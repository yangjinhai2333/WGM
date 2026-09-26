"""Durable candidate preparation/reconciliation for validated operations only."""
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import stat
import tempfile

import evaluation_ledger as io
from evaluation_plan import durable_json
import candidate_workspace as shared
from candidate_provenance import (ROOT, ProvenanceError, read_chain, require_candidate,
                                  source_digest, validate_metadata, with_provenance)
from search_operation_journal import require, atomic_noreplace, durable_record


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sync_tree(root):
    """Never follow a model-controlled link, including Git management files."""
    io._checked_path(root)
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            if (base / name).is_symlink():
                raise shared.CandidateError("Unsafe workspace link")
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise shared.CandidateError("Unsafe workspace file")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        io._sync_dir(base)


def verify_source_links(root):
    io._checked_path(root)
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            io._checked_path(Path(directory) / name)
        for name in names:
            io._regular(Path(directory) / name)


def immutable_bytes(path, data):
    io._checked_path(path)
    if path.exists():
        io._regular(path)
        require(path.read_bytes() == data, "BLOCK_IDENTITY_MISMATCH")
        return
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_noreplace(temporary, path)
    io._sync_dir(path.parent)


def git_management(workspace):
    """Index/objects can change; refs, config and every other control file cannot."""
    root = workspace / ".git"
    values = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if name == "index" or name == "objects" or name.startswith("objects/"):
            continue
        values[name] = sha(path.read_bytes()) if path.is_file() else "directory"
    return io.digest(values)


class OperationWorkspace:
    def __init__(self, run_directory, sources, config, *, fault=lambda point: None, prompt_provider=None):
        self.directory, self.sources, self.config = Path(run_directory), sources, config
        self.fault = fault
        self.prompt_provider = prompt_provider  # None: the original Codex prompt text.

    def paths(self, session):
        base = session.journal.folder(session.plan) / "workspace" / "prepared"
        return base, base / "agent", base / "parent"

    def parent(self, plan):
        identity = plan.identity["parent"]
        source = self.sources.get(identity["candidate_id"])
        verify_source_links(source["path"])
        require(source["source_fingerprint"] == identity["source_digest"]
                and source["chain_digest"] == identity["provenance_digest"], "BLOCK_IDENTITY_MISMATCH")
        chain = read_chain(self.directory, identity["candidate_id"], require_versioned=True)
        for item in chain:
            folder = self.directory / item.provenance.candidate_id
            io._regular(folder / "metadata.json")
            io._regular(folder / "model_patch.diff")
        return source, chain[-1].provenance if chain else ROOT

    def prepare(self, session):
        session.check()
        source, _ = self.parent(session.plan)
        base, workspace, parent = self.paths(session)
        if not base.exists():
            # No published state refers to this temporary tree. A crash can
            # rebuild it without replaying a patch into an existing workspace.
            with tempfile.TemporaryDirectory(prefix=".tmp-", dir=base.parent) as temporary:
                bundle = Path(temporary) / "prepared"
                bundle.mkdir()
                agent, verifier = bundle / "agent", bundle / "parent"
                shared._copy_source(source["path"], agent, False,
                    preserve_source=(session.plan.identity["algorithm"] in ("wgm_v2", "dgm") or
                        getattr(self.sources, 'preserve_source', False)))
                require(source_digest(agent) == session.plan.identity["parent"]["source_digest"],
                        "BLOCK_IDENTITY_MISMATCH")
                shared._git(agent, "init", "-q", "--template=")
                shared._check_workspace(agent)
                shared._git(agent, "add", "--all", "--force", "--", ".")
                # A Git tree is a deterministic baseline: no author/date/PID.
                baseline = shared._git(agent, "write-tree").decode().strip()
                shutil.copytree(agent, verifier, ignore=shutil.ignore_patterns(".git"))
                shared._git(verifier, "init", "-q", "--template=")
                descriptor = {"baseline_tree": baseline, "parent_source": source_digest(verifier),
                    "git_config": sha((agent / ".git/config").read_bytes()),
                    "git_head": sha((agent / ".git/HEAD").read_bytes()), "git_management": git_management(agent)}
                durable_record(bundle / "baseline.json", descriptor)
                sync_tree(bundle)
                self.fault("workspace.before_publish")
                atomic_noreplace(bundle, base)
                io._sync_dir(base.parent)
                self.fault("workspace.after_publish")
        descriptor = io._read_json(base / "baseline.json")
        io._keys(descriptor, ("baseline_tree", "parent_source", "git_config", "git_head", "git_management"))
        require(descriptor["parent_source"] == session.plan.identity["parent"]["source_digest"],
                "BLOCK_IDENTITY_MISMATCH")
        artifact = session.artifact("workspace.json", {"kind": "workspace", **descriptor})
        if session.state["workspace"] is None:
            session.append("workspace_ready", {"artifact": artifact})
        else:
            require(artifact == session.state["workspace"], "BLOCK_IDENTITY_MISMATCH")
        self.verify(session, pristine=True)
        return workspace

    def verify(self, session, *, pristine=False):
        self.parent(session.plan)
        base, workspace, parent = self.paths(session)
        descriptor = session.journal.read_artifact(session.plan, session.state["workspace"])
        require(io._read_json(base / "baseline.json") == {k: descriptor[k] for k in
            ("baseline_tree", "parent_source", "git_config", "git_head", "git_management")}, "BLOCK_IDENTITY_MISMATCH")
        sync_tree(base)
        for name, key in (("config", "git_config"), ("HEAD", "git_head")):
            path = workspace / ".git" / name
            if not path.is_file() or sha(path.read_bytes()) != descriptor[key]:
                raise shared.CandidateError("Candidate changed Git management files")
        if git_management(workspace) != descriptor["git_management"]:
            raise shared.CandidateError("Candidate changed Git management files")
        shared._check_workspace(workspace)
        require(source_digest(parent) == descriptor["parent_source"], "BLOCK_IDENTITY_MISMATCH")
        if pristine:
            require(source_digest(workspace) == descriptor["parent_source"], "BLOCK_IDENTITY_MISMATCH")
        return descriptor

    def prompt(self, session, problem):
        from prompts.testrepo_prompt import get_test_description
        _, workspace, _ = self.paths(session)
        test_description = get_test_description(swerepo=False)
        prompt = (
            f"Modify only the candidate workspace {workspace} to solve this self-improvement problem.\n"
            f"<problem_description>\n{problem}\n</problem_description>\n\n"
            f"<test_description>\n{test_description.replace('/hgm/', shlex.quote(str(workspace) + '/'))}\n</test_description>\n\n"
            "You may inspect files and run lightweight tests. Do not invoke a model, Docker, "
            "SWE-bench or Polyglot evaluations. Do not install dependencies. "
            "Do not modify benchmark data, outputs, credentials "
            "or Git management files. Do not create commits. Avoid unrelated changes. "
            "The result is collected from Git diff, including new and deleted files; "
            "do not return a textual patch. Briefly report the changes and tests."
        )
        # Check the frozen parent, even before workspace preparation.
        source, _ = self.parent(session.plan)
        if (source["path"] / "prompts/downstream_instructions.txt").is_file():
            from provider_prompts import policy_note
            prompt += policy_note(self.prompt_provider)
        return prompt

    def snapshot(self, session):
        """Terminal-owned safe evidence, not stdout or model claims."""
        # Corrupt parent provenance is never evidence against the model child.
        self.parent(session.plan)
        try:
            self.verify(session)
            return {"kind": "generation_result", "valid_workspace": True,
                    "source_digest": source_digest(self.paths(session)[1])}
        except (shared.CandidateError, ProvenanceError):
            return {"kind": "generation_result", "valid_workspace": False, "source_digest": None}

    def seal(self, session, terminal_artifact):
        evidence = session.journal.read_artifact(session.plan, terminal_artifact)
        require(evidence["kind"] == "generation_result")
        if not evidence["valid_workspace"]:
            return self.invalid(session, "WORKSPACE_INVALID")
        descriptor = self.verify(session)
        _, workspace, parent = self.paths(session)
        require(source_digest(workspace) == evidence["source_digest"], "BLOCK_IDENTITY_MISMATCH")
        _, parent_provenance = self.parent(session.plan)
        cfg = self.config
        try:
            if cfg.validation_command:
                code, expired, _, _ = shared._execute(cfg.validation_command, workspace,
                    shared._clean_environment(), b"", cfg.validation_timeout_seconds)
                if expired:
                    raise OSError("VALIDATION_IO")
                if code != 0:
                    return self.invalid(session, "TESTS_FAILED")
            self.verify(session)
            # Tests may create excluded caches but cannot rewrite candidate source.
            require(source_digest(workspace) == evidence["source_digest"], "BLOCK_IDENTITY_MISMATCH")
            excludes = [f":(glob,exclude)**/{name}/**" for name in sorted(shared._CACHES)]
            shared._git(workspace, "add", "--all", "--force", "--", ".", *excludes)
            patch = shared._git(workspace, "diff", "--cached", "--no-ext-diff", "--no-textconv",
                "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", descriptor["baseline_tree"], "--", ".", *excludes)
            if len(patch) > cfg.max_patch_bytes:
                return self.invalid(session, "PATCH_TOO_LARGE")
            paths = shared._patch_paths(workspace, patch)
            for name in paths:
                path = workspace / name
                if path.suffix == ".py" and path.is_file():
                    compile(path.read_bytes(), name, "exec")
            # Verify the actual resulting source, not just applicability or
            # the model-controlled Git index/objects. This fresh copy already
            # contains each ancestor once; only the self patch is applied here.
            with tempfile.TemporaryDirectory(prefix=".tmp-verify-", dir=parent.parent.parent) as temporary:
                replayed = Path(temporary) / "agent"
                shutil.copytree(parent, replayed, ignore=shutil.ignore_patterns(".git"))
                shared._git(replayed, "init", "-q", "--template=")
                shared._git(replayed, "apply", "--check", "-p1", input=patch)
                shared._git(replayed, "apply", "-p1", input=patch)
                shared._check_workspace(replayed)
                if source_digest(replayed) != evidence["source_digest"]:
                    raise shared.CandidateError("Patch does not reproduce candidate source")
            metadata = with_provenance({"parent_commit": session.plan.identity["parent"]["candidate_id"],
                "status": "success", "stage": "complete", "backend": "codex_exec",
                "downstream_backend": "codex_exec",
                "operation_id": session.plan.operation_id, "entry": session.plan.identity["diagnosis"]["entry"],
                "overall_performance": {"accuracy_score": 0.0, "total_resolved_instances": 0,
                    "total_submitted_instances": 0, "files": [], "total_submitted_ids": [],
                    "total_unresolved_ids": [], "total_emptypatch_ids": [], "total_resolved_ids": []}},
                session.plan.identity["candidate_id"], parent_provenance, patch,
                descriptor["parent_source"], evidence["source_digest"])
        except shared.CandidateError as error:
            # The shared transport intentionally hides Git stderr. A generic
            # Git failure cannot prove an invalid candidate (e.g. disk full).
            if str(error) == "Git operation failed":
                raise OSError("VALIDATION_IO") from None
            return self.invalid(session, "PATCH_INVALID")
        except (ProvenanceError, SyntaxError, UnicodeError):
            return self.invalid(session, "PATCH_INVALID")
        folder = session.journal.folder(session.plan) / "candidate"
        io._mkdir(folder)
        immutable_bytes(folder / "model_patch.diff", patch)
        immutable_bytes(folder / "metadata.json", io.canonical(metadata) + b"\n")
        validate_metadata(metadata, session.plan.identity["candidate_id"], parent_provenance, patch)
        artifact = session.artifact("candidate-seal.json", {"kind": "candidate", "metadata": metadata,
            "patch_digest": sha(patch), "candidate_digest": io.digest(metadata["provenance"])})
        self.fault("generation.after_seal")
        return {"outcome": "candidate", "artifact": artifact, "candidate_digest": io.digest(metadata["provenance"])}

    def invalid(self, session, code):
        artifact = session.artifact("candidate-invalid.json", {"kind": "candidate_invalid", "code": code})
        return {"outcome": "candidate_invalid", "artifact": artifact, "candidate_digest": artifact["sha256"]}

    def publish(self, session, terminal, *, committed=False):
        session.check()
        if terminal["outcome"] != "candidate":
            return
        artifact = session.journal.read_artifact(session.plan, terminal["artifact"])
        require(artifact["kind"] == "candidate" and artifact["candidate_digest"] == terminal["candidate_digest"])
        cid = session.plan.identity["candidate_id"]
        target = io._checked_path(self.directory / cid)
        if not target.exists():
            folder = session.journal.folder(session.plan) / "candidate"
            self._verify_candidate(session, folder, artifact, strict=True)
            sync_tree(folder)
            self.fault("candidate.before_publish")
            # Run and operation leases exclude other authorized publishers.
            atomic_noreplace(folder, target)
            io._sync_dir(self.directory)
            io._sync_dir(folder.parent)
            self.fault("candidate.after_publish")
        self._verify_candidate(session, target, artifact, strict=not committed)
        chain = require_candidate(self.directory, cid, session.plan.identity["parent"]["candidate_id"])
        source = self.sources.get(cid)
        verify_source_links(source["path"])
        require(source["source_fingerprint"] == chain[-1].provenance.candidate_source_sha256
                and source["chain_digest"] == io.digest([item.provenance.to_dict() for item in chain]),
                "BLOCK_IDENTITY_MISMATCH")

    def _verify_candidate(self, session, folder, artifact, *, strict):
        if strict:
            require({p.name for p in folder.iterdir()} == {"metadata.json", "model_patch.diff"},
                    "BLOCK_IDENTITY_MISMATCH")
        io._regular(folder / "metadata.json")
        io._regular(folder / "model_patch.diff")
        metadata = io._read_json(folder / "metadata.json")
        patch = (folder / "model_patch.diff").read_bytes()
        require(sha(patch) == artifact["patch_digest"], "BLOCK_IDENTITY_MISMATCH")
        expected = artifact["metadata"]
        # Evaluation views may change only the derived performance fields.
        require((io.same_json(metadata, expected) if strict else io.same_json(
                 {k: v for k, v in metadata.items() if k != "overall_performance"},
                 {k: v for k, v in expected.items() if k != "overall_performance"})), "BLOCK_IDENTITY_MISMATCH")
        _, parent = self.parent(session.plan)
        provenance = validate_metadata(metadata, session.plan.identity["candidate_id"], parent, patch)
        require(io.digest(provenance.to_dict()) == artifact["candidate_digest"], "BLOCK_IDENTITY_MISMATCH")
