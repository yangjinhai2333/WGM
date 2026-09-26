"""Immutable candidate provenance, shared by generation and replay readers.

Legacy records are accepted only with an explicit parent and matching self and
ancestor hashes. Reading never repairs metadata or writes into a run directory.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re


class ProvenanceError(ValueError):
    """A fixed, non-sensitive error; never include candidate-controlled content."""

    def __init__(self, message, *, code="PROV_INVALID"):
        self.code = code
        super().__init__(f"{code}: {message}")


def require_candidate_backend(backend):
    if backend != "codex_exec":
        raise ProvenanceError("Candidate generation requires codex_exec with versioned provenance; "
                              "legacy_api has no verified source snapshots",
                              code="PROV_BACKEND_UNSUPPORTED")


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _id(value):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or value in (".", "..") or "\\" in value):
        raise ProvenanceError("Invalid candidate identity", code="PROV_ID_INVALID")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProvenanceError("Missing or invalid provenance digest", code="PROV_DIGEST_INVALID")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError("Ambiguous duplicate metadata field", code="PROV_JSON_DUPLICATE")
        result[key] = value
    return result


@dataclass(frozen=True)
class PatchRef:
    candidate_id: str
    patch_sha256: str

    def __post_init__(self):
        _id(self.candidate_id)
        _sha(self.patch_sha256)
        if self.candidate_id == "initial":
            raise ProvenanceError("Initial candidate cannot have a self patch", code="PROV_ROOT_PATCH")


@dataclass(frozen=True)
class Provenance:
    candidate_id: str
    parent_candidate_id: str | None
    ancestor_chain: tuple[PatchRef, ...]
    self_patch: PatchRef | None
    parent_source_sha256: str | None = None
    candidate_source_sha256: str | None = None

    def __post_init__(self):
        _id(self.candidate_id)
        if type(self.ancestor_chain) is not tuple or any(type(p) is not PatchRef for p in self.ancestor_chain):
            raise ProvenanceError("Ancestor chain must be immutable PatchRefs", code="PROV_CHAIN_INVALID")
        if self.candidate_id == "initial":
            if (self.parent_candidate_id is not None or self.ancestor_chain or self.self_patch is not None
                    or self.parent_source_sha256 is not None or self.candidate_source_sha256 is not None):
                raise ProvenanceError("Initial candidate must be patchless", code="PROV_ROOT_PATCH")
            return
        _id(self.parent_candidate_id)
        if type(self.self_patch) is not PatchRef or self.self_patch.candidate_id != self.candidate_id:
            raise ProvenanceError("Candidate self patch identity mismatch", code="PROV_SELF_MISMATCH")
        expected_parent = self.ancestor_chain[-1].candidate_id if self.ancestor_chain else "initial"
        if self.parent_candidate_id != expected_parent:
            raise ProvenanceError("Candidate parent mismatch", code="PROV_PARENT_MISMATCH")
        chain = self.ancestor_chain + (self.self_patch,)
        if (len({p.candidate_id for p in chain}) != len(chain)
                or len({p.patch_sha256 for p in chain}) != len(chain)):
            raise ProvenanceError("Duplicate self patch in complete chain", code="PROV_DUPLICATE_PATCH")
        if self.parent_source_sha256 is not None or self.candidate_source_sha256 is not None:
            _sha(self.parent_source_sha256)
            _sha(self.candidate_source_sha256)

    @property
    def complete_chain(self):
        return self.ancestor_chain + ((self.self_patch,) if self.self_patch else ())

    def to_dict(self):
        ancestors = [asdict(p) for p in self.ancestor_chain]
        complete = [asdict(p) for p in self.complete_chain]
        body = {"schema_version": 1, "candidate_id": self.candidate_id,
                "parent_candidate_id": self.parent_candidate_id,
                "ancestor_chain": ancestors,
                "self_patch": asdict(self.self_patch) if self.self_patch else None,
                "complete_chain": complete, "ancestor_chain_sha256": _digest(ancestors),
                "complete_chain_sha256": _digest(complete),
                "parent_source_sha256": self.parent_source_sha256,
                "candidate_source_sha256": self.candidate_source_sha256}
        return {**body, "sha256": _digest(body)}


ROOT = Provenance("initial", None, (), None)


@dataclass(frozen=True)
class ValidatedCandidate:
    provenance: Provenance
    metadata_sha256: str
    patch_bytes: bytes


def validate_metadata(metadata, candidate_id, parent, patch_bytes):
    """Pure reader/writer validator; derive complete chain only from parent+self.

    The parent is an already validated immutable representation. Legacy fields
    remain checked aliases, never alternate interpretations of a versioned record.
    """
    _id(candidate_id)
    if candidate_id == "initial" or not isinstance(metadata, dict):
        raise ProvenanceError("Invalid non-root candidate metadata", code="PROV_METADATA_INVALID")
    if metadata.get("parent_commit") != parent.candidate_id:
        raise ProvenanceError("Candidate parent mismatch", code="PROV_PARENT_MISMATCH")
    if "run_id" in metadata and metadata["run_id"] != candidate_id:
        raise ProvenanceError("Candidate identity mismatch", code="PROV_ID_MISMATCH")
    if "status" in metadata and metadata["status"] != "success":
        raise ProvenanceError("Candidate is not successfully published", code="PROV_NOT_CANDIDATE")
    if not isinstance(patch_bytes, bytes) or not patch_bytes.strip():
        raise ProvenanceError("Required candidate patch is empty", code="PROV_PATCH_EMPTY")
    patch_hash = hashlib.sha256(patch_bytes).hexdigest()
    if _sha(metadata.get("patch_sha256")) != patch_hash:
        raise ProvenanceError("Candidate patch hash changed", code="PROV_PATCH_HASH_MISMATCH")
    ancestors = parent.complete_chain
    if metadata.get("ancestor_patch_sha256") != [p.patch_sha256 for p in ancestors]:
        raise ProvenanceError("Ancestor-only chain mismatch; ambiguous legacy metadata is not repaired",
                              code="PROV_ANCESTOR_MISMATCH")
    if "parent_patch_count" in metadata and (type(metadata["parent_patch_count"]) is not int
                                           or metadata["parent_patch_count"] != len(ancestors)):
        raise ProvenanceError("Candidate ancestor count mismatch", code="PROV_ANCESTOR_COUNT")
    if (candidate_id in {p.candidate_id for p in ancestors}
            or patch_hash in {p.patch_sha256 for p in ancestors}):
        raise ProvenanceError("Duplicate self patch in complete chain", code="PROV_DUPLICATE_PATCH")
    versioned = any(k in metadata for k in ("candidate_schema_version", "provenance", "artifact_type"))
    parent_source = candidate_source = None
    if versioned:
        saved = metadata.get("provenance")
        if (type(metadata.get("candidate_schema_version")) is not int
                or metadata["candidate_schema_version"] != 1 or not isinstance(saved, dict)
                or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1):
            raise ProvenanceError("Unsupported or incomplete candidate provenance schema", code="PROV_SCHEMA_UNSUPPORTED")
        if (metadata.get("artifact_type") != "candidate" or metadata.get("status") != "success"
                or metadata.get("run_id") != candidate_id or "parent_patch_count" not in metadata):
            raise ProvenanceError("Incomplete candidate schema identity", code="PROV_SCHEMA_INCOMPLETE")
        parent_source = _sha(saved.get("parent_source_sha256"))
        candidate_source = _sha(saved.get("candidate_source_sha256"))
        if parent.candidate_source_sha256 and parent_source != parent.candidate_source_sha256:
            raise ProvenanceError("Parent source snapshot mismatch", code="PROV_PARENT_SOURCE_MISMATCH")
    result = Provenance(candidate_id, parent.candidate_id, ancestors,
                        PatchRef(candidate_id, patch_hash), parent_source, candidate_source)
    if versioned and metadata["provenance"] != result.to_dict():
        raise ProvenanceError("Candidate provenance schema or digest mismatch", code="PROV_SCHEMA_DIGEST_MISMATCH")
    return result


def with_provenance(metadata, candidate_id, parent, patch_bytes, parent_source, candidate_source):
    """Return a fresh versioned metadata object; do not alias caller lists."""
    result = deepcopy(metadata)
    result.setdefault("run_id", candidate_id)
    result.update(candidate_schema_version=1, artifact_type="candidate",
                  ancestor_patch_sha256=[p.patch_sha256 for p in parent.complete_chain],
                  parent_patch_count=len(parent.complete_chain),
                  patch_sha256=hashlib.sha256(patch_bytes).hexdigest())
    result["provenance"] = Provenance(candidate_id, parent.candidate_id, parent.complete_chain,
        PatchRef(candidate_id, result["patch_sha256"]), parent_source, candidate_source).to_dict()
    validate_metadata(result, candidate_id, parent, patch_bytes)
    return result


def read_chain(run_dir, candidate_id, *, require_versioned=False):
    """Return validated root-to-candidate records, excluding the patchless root."""
    run_dir = Path(run_dir).resolve()
    pending, seen = [], set()
    while candidate_id != "initial":
        _id(candidate_id)
        if candidate_id in seen:
            raise ProvenanceError("Parent patch chain contains a cycle", code="PROV_PARENT_CYCLE")
        seen.add(candidate_id)
        node = run_dir / candidate_id
        metadata_path, patch_path = node / "metadata.json", node / "model_patch.diff"
        if node.is_symlink() or metadata_path.is_symlink() or patch_path.is_symlink():
            raise ProvenanceError("Candidate provenance must stay inside this run", code="PROV_PATH_INVALID")
        try:
            raw = metadata_path.read_bytes()
            metadata = json.loads(raw, object_pairs_hook=_unique_object)
            data = patch_path.read_bytes()
        except ProvenanceError:
            raise
        except (OSError, ValueError):
            raise ProvenanceError("Required candidate metadata or patch is missing or unreadable",
                                  code="PROV_ARTIFACT_UNREADABLE") from None
        if not isinstance(metadata, dict) or "parent_commit" not in metadata:
            raise ProvenanceError("Legacy metadata has no explicit parent; refusing to guess", code="PROV_PARENT_MISSING")
        if require_versioned and not pending and metadata.get("candidate_schema_version") != 1:
            raise ProvenanceError("Search admission/evaluation requires versioned candidate metadata",
                                  code="PROV_SCHEMA_REQUIRED")
        pending.append((candidate_id, metadata, data, hashlib.sha256(raw).hexdigest()))
        candidate_id = _id(metadata["parent_commit"])
    validate_root(run_dir / "initial")
    result, parent = [], ROOT
    for candidate_id, metadata, data, metadata_hash in reversed(pending):
        parent = validate_metadata(metadata, candidate_id, parent, data)
        result.append(ValidatedCandidate(parent, metadata_hash, data))
    return tuple(result)


def validate_root(root):
    """The initial source (including a named initial cache) has no patch."""
    root = Path(root)
    if root.is_symlink() or (root / "metadata.json").is_symlink():
        raise ProvenanceError("Initial provenance must stay inside this run", code="PROV_PATH_INVALID")
    if (root / "model_patch.diff").exists() or (root / "model_patch.diff").is_symlink():
        raise ProvenanceError("Initial candidate cannot have a self patch", code="PROV_ROOT_PATCH")
    if (root / "metadata.json").is_file():
        try:
            metadata = json.loads((root / "metadata.json").read_bytes(), object_pairs_hook=_unique_object)
        except ProvenanceError:
            raise
        except (OSError, ValueError):
            raise ProvenanceError("Initial metadata is unreadable", code="PROV_ARTIFACT_UNREADABLE") from None
        if (not isinstance(metadata, dict) or metadata.get("parent_commit") is not None
                or metadata.get("patch_sha256") is not None
                or metadata.get("ancestor_patch_sha256", []) != []
                or type(metadata.get("parent_patch_count", 0)) is not int
                or metadata.get("parent_patch_count", 0) != 0):
            raise ProvenanceError("Initial candidate must be patchless", code="PROV_ROOT_PATCH")
        if ("candidate_schema_version" in metadata or "provenance" in metadata) and (
                type(metadata.get("candidate_schema_version")) is not int
                or metadata["candidate_schema_version"] != 1 or metadata.get("provenance") != ROOT.to_dict()
                or type(metadata["provenance"]["schema_version"]) is not int):
            raise ProvenanceError("Initial provenance schema must describe the patchless root", code="PROV_ROOT_SCHEMA")


def source_digest(directory):
    """Digest replayable files, including executable bits; exclude runtime caches."""
    from candidate_workspace import _forbidden
    directory = Path(directory)
    if not directory.is_dir():
        raise ProvenanceError("Source snapshot directory is missing", code="PROV_SOURCE_MISSING")
    files = {}
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        if _forbidden(name):
            continue
        if path.is_symlink():
            raise ProvenanceError("Source snapshot contains a symlink", code="PROV_SOURCE_SYMLINK")
        if path.is_file():
            files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "executable": bool(path.stat().st_mode & 0o111)}
    return _digest(files)


def replay(workspace, chain):
    """Apply each validated patch once and verify available source snapshots."""
    from candidate_workspace import _git, _patch_paths, _check_workspace
    chain = tuple(chain)
    parent = ROOT
    # Reject reordered, duplicated or hand-constructed records before applying any patch.
    for item in chain:
        if (item.provenance.ancestor_chain != parent.complete_chain
                or hashlib.sha256(item.patch_bytes).hexdigest() != item.provenance.self_patch.patch_sha256):
            raise ProvenanceError("Replay chain order or patch bytes changed", code="PROV_REPLAY_CHAIN")
        parent = item.provenance
    for item in chain:
        provenance = item.provenance
        if (provenance.parent_source_sha256 is not None
                and source_digest(workspace) != provenance.parent_source_sha256):
            raise ProvenanceError("Replay parent source snapshot mismatch", code="PROV_REPLAY_PARENT_SOURCE")
        _patch_paths(workspace, item.patch_bytes)
        _git(workspace, "apply", "--check", "-p1", input=item.patch_bytes)
        _git(workspace, "apply", "-p1", input=item.patch_bytes)
        _check_workspace(workspace)
        if (provenance.candidate_source_sha256 is not None
                and source_digest(workspace) != provenance.candidate_source_sha256):
            raise ProvenanceError("Replay candidate source snapshot mismatch", code="PROV_REPLAY_CANDIDATE_SOURCE")


def require_candidate(run_dir, candidate_id, parent_candidate_id):
    """Read-only gate for creating a search node or consuming model evaluations."""
    if candidate_id == "initial":
        raise ProvenanceError("Initial is not a generated child", code="PROV_NOT_CANDIDATE")
    chain = read_chain(run_dir, candidate_id, require_versioned=True)
    if chain[-1].provenance.parent_candidate_id != parent_candidate_id:
        raise ProvenanceError("Candidate parent differs from search tree", code="PROV_PARENT_MISMATCH")
    return chain


def validate_search_nodes(run_dir, rows, pending=None):
    """Validate disk identity before existing restore/reconcile constructs nodes.

    No checkpoint writes, RNG draws, recovery decisions or state changes occur here.
    """
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len({r["commit_id"] for r in rows}) != len(rows):
        raise ProvenanceError("Duplicate search node identity", code="PROV_NODE_DUPLICATE")
    root = by_id.get(0)
    if root is None or root["commit_id"] != "initial" or root["parent_id"] is not None:
        raise ProvenanceError("Search root identity mismatch", code="PROV_ROOT_SCHEMA")
    validate_root(Path(run_dir) / "initial")
    for row in rows:
        if row["id"] == 0:
            continue
        if row["parent_id"] not in by_id:
            raise ProvenanceError("Search node has an unknown parent", code="PROV_PARENT_MISSING")
        require_candidate(run_dir, row["commit_id"], by_id[row["parent_id"]]["commit_id"])
    if pending is not None:
        path = Path(run_dir) / _id(pending["candidate_id"]) / "metadata.json"
        if path.exists():
            try:
                metadata = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
            except ProvenanceError:
                raise
            except (OSError, ValueError):
                raise ProvenanceError("Pending metadata is unreadable", code="PROV_ARTIFACT_UNREADABLE") from None
            if not isinstance(metadata, dict):
                raise ProvenanceError("Pending metadata is invalid", code="PROV_METADATA_INVALID")
            if metadata.get("status") == "success":
                if pending["parent_id"] not in by_id:
                    raise ProvenanceError("Pending candidate has an unknown parent", code="PROV_PARENT_MISSING")
                require_candidate(run_dir, pending["candidate_id"], by_id[pending["parent_id"]]["commit_id"])
