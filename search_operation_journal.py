"""Independent, append-only search operation journal. No task evaluation records."""
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import math
import os
import re
import stat
import tempfile
from pathlib import Path

import evaluation_ledger as io
from evaluation_plan import durable_json

SCHEMA = 1
PLAN_FIELDS = frozenset(("schema_version", "run_id", "ordinal", "algorithm", "profile", "policy",
    "run_fingerprint", "parent", "candidate_id", "diagnosis", "generation", "fingerprints",
    "budget", "pre_rng", "post_rng", "pre_rng_digest", "post_rng_digest", "checkpoint_digest"))


class OperationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(ok, code="BLOCK_CORRUPT_OPERATION"):
    if not ok:
        raise OperationError(code)


def atomic_noreplace(source, target):
    """Linux atomic, no-clobber publication without a transient hardlink.

    No copy or overwriting-rename fallback: unsupported filesystems fail closed.
    Callers own the run/operation lease and fsync data before, directories after.
    """
    source, target = io._checked_path(source), io._checked_path(target)
    if source.is_file():
        io._regular(source)
    else:
        require(source.is_dir(), "BLOCK_CORRUPT_OPERATION")
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "OPERATION_ATOMIC_PUBLISH_UNSUPPORTED")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        raise OSError(ctypes.get_errno(), "OPERATION_ATOMIC_PUBLISH_FAILED")


def durable_record(path, value, *, fault=lambda point: None):
    """Immutable JSON publication. Heads/checkpoints use replaceable durable_json."""
    path = io._checked_path(path)
    data = io.canonical(value) + b"\n"
    require(len(data) <= io.MAX_JSON_BYTES, "BLOCK_CORRUPT_OPERATION")
    fault("checkpoint.before_write")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        fault("checkpoint.after_write")
        os.fsync(stream.fileno())
        fault("checkpoint.after_file_fsync")
    fault("checkpoint.before_publish")
    atomic_noreplace(temporary, path)
    fault("checkpoint.after_rename")
    io._sync_dir(path.parent)
    fault("checkpoint.after_dir_fsync")
    fault("checkpoint.after_publish")


def validate_rng(value):
    """RNG schema 1: exact MT19937 arrays, no coercion or RNG side effects."""
    if type(value) is dict and type(value.get("schema_version")) is int and value["schema_version"] == 2:
        from wgm_v2_selector import validate_python_rng
        io._keys(value, ("schema_version", "algorithm", "selection", "task"))
        require(value["algorithm"] == "wgm_v2")
        validate_python_rng(value["selection"])
        validate_python_rng(value["task"])
        return
    io._keys(value, ("schema_version", "global", "selector"))
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    io._keys(value["global"], ("python", "numpy"))
    def integer(v, maximum):
        return type(v) is int and 0 <= v <= maximum
    def words(row, length):
        require(type(row) is list and len(row) == length)
        require(all(integer(v, 2 ** 32 - 1) for v in row[:624]))
    def finite(v):
        return type(v) is float and math.isfinite(v)
    def numpy_state(row):
        require(type(row) is list and len(row) == 5 and row[0] == "MT19937")
        words(row[1], 624)
        require(integer(row[2], 624) and integer(row[3], 1) and finite(row[4]))
    python = value["global"]["python"]
    require(type(python) is list and len(python) == 3
            and type(python[0]) is int and python[0] == 3)
    words(python[1], 625)
    require(integer(python[1][624], 624) and (python[2] is None or finite(python[2])))
    numpy_state(value["global"]["numpy"])
    if value["selector"] is not None:
        io._keys(value["selector"], ("mode", "state"))
        require(value["selector"]["mode"] in ("global_numpy", "private_numpy"))
        numpy_state(value["selector"]["state"])
        require(value["selector"]["mode"] != "global_numpy" or
                value["selector"]["state"] == value["global"]["numpy"])


@dataclass(frozen=True)
class SearchOperationPlan:
    """Canonical bytes prevent mutation through nested dictionaries/lists."""
    encoded: bytes

    def __post_init__(self):
        row = io._decode(self.encoded)
        io._keys(row, ("operation_id", "identity"))
        body = row["identity"]
        io._keys(body, PLAN_FIELDS)
        require(type(body["schema_version"]) is int and body["schema_version"] == SCHEMA)
        require(row["operation_id"] == "op-" + io.digest(body), "BLOCK_IDENTITY_MISMATCH")
        require(body["policy"] == "validated_observation_v2")
        require((body["algorithm"], body["profile"]) in (("hgm","legacy_code"),
            ("hgm","paper_corrected"),("wgm","paper_corrected"),("wgm_v2","wgm_exp3_bai_v2"),("dgm","legacy_code")))
        require(type(body["ordinal"]) is int and body["ordinal"] >= 0)
        require(body["candidate_id"] == f"candidate-{body['ordinal']:06d}")
        for key in ("run_id", "candidate_id"): io._id(body[key])
        for key in ("run_fingerprint", "checkpoint_digest", "pre_rng_digest", "post_rng_digest"):
            io._sha(body[key])
        require(io.digest(body["pre_rng"]) == body["pre_rng_digest"]
                and io.digest(body["post_rng"]) == body["post_rng_digest"], "BLOCK_IDENTITY_MISMATCH")
        validate_rng(body["pre_rng"])
        validate_rng(body["post_rng"])
        require((body["pre_rng"]["schema_version"] == body["post_rng"]["schema_version"] == 2)
                if body["algorithm"] == "wgm_v2" else
                (body["pre_rng"]["schema_version"] == body["post_rng"]["schema_version"] == 1))
        io._keys(body["parent"], ("node_id","candidate_id","branch_id","provenance_digest","source_digest"))
        io._id(body["parent"]["candidate_id"])
        for key in ("node_id", "branch_id"):
            require(type(body["parent"][key]) is int and body["parent"][key] >= 0)
        for key in ("provenance_digest","source_digest"): io._sha(body["parent"][key])
        io._keys(body["diagnosis"], ("entry","task_id","entry_digest","commit","base_commit","prompt_digest","max_attempts"))
        require(type(body["diagnosis"]["max_attempts"]) is int and body["diagnosis"]["max_attempts"] == (4 if body["algorithm"] == "dgm" else 3))
        if body["diagnosis"]["base_commit"] is not None:
            io._id(body["diagnosis"]["base_commit"])
        require(body["diagnosis"]["commit"] == body["parent"]["candidate_id"])
        for key in ("entry","task_id"): io._id(body["diagnosis"][key])
        for key in ("entry_digest","prompt_digest"): io._sha(body["diagnosis"][key])
        io._keys(body["generation"], ("prompt_recipe_digest",))
        io._sha(body["generation"]["prompt_recipe_digest"])
        io._keys(body["fingerprints"], ("configuration","model","policy","source","builder","schema","task_schedule"))
        for value in body["fingerprints"].values(): io._sha(value)
        require(body["fingerprints"]["source"] == body["parent"]["source_digest"]
                and body["parent"]["candidate_id"] != body["candidate_id"], "BLOCK_IDENTITY_MISMATCH")
        io._keys(body["budget"], ("used", "limit", "initial_attempts", "initialized", "expansion"))
        for key in ("used", "limit", "initial_attempts"):
            require(type(body["budget"][key]) is int and body["budget"][key] >= 0)
        require(body["budget"]["used"] <= body["budget"]["limit"] and type(body["budget"]["initialized"]) is bool)
        expansion = body["budget"]["expansion"]
        io._keys(expansion, ("expansion_id", "candidate_id", "parent_id", "selected_branch_id",
                            "selection_probability", "child_id", "phase", "target_M", "status"))
        require(expansion["candidate_id"] == body["candidate_id"] and expansion["status"] == "creating"
                and expansion["expansion_id"] == f"expansion-{body['ordinal']:06d}"
                and expansion["parent_id"] == body["parent"]["node_id"]
                and expansion["selected_branch_id"] == body["parent"]["branch_id"]
                and expansion["child_id"] is None and expansion["phase"] in ("initialization", "search")
                and type(expansion["target_M"]) is int and expansion["target_M"] > 0)
        probability = expansion["selection_probability"]
        if body["algorithm"] in ("wgm", "wgm_v2") and expansion["phase"] == "search":
            require(type(probability) is float and math.isfinite(probability) and 0 < probability <= 1)
        else:
            require(probability is None)
        require(self.encoded == io.canonical(row))

    @classmethod
    def create(cls, **body):
        body = {"schema_version": SCHEMA, **body}
        return cls.from_dict({"operation_id":"op-"+io.digest(body),"identity":body})

    @classmethod
    def from_dict(cls, row):
        return cls(io.canonical(row))

    def to_dict(self): return io._decode(self.encoded)
    @property
    def identity(self): return self.to_dict()["identity"]
    @property
    def operation_id(self): return self.to_dict()["operation_id"]
    @property
    def digest(self): return hashlib.sha256(self.encoded).hexdigest()


def call_id(plan, stage, attempt):
    return io.digest({"operation_id":plan.operation_id,"stage":stage,"attempt":attempt,"schema_version":SCHEMA})


def _terminal(value):
    io._keys(value, ("started","reaped","returncode","launch_failed","timed_out","interrupted"))
    io.validate_execution_evidence({**value, "duration_seconds":0, "input_tokens":None,
                                   "cached_input_tokens":None,"output_tokens":None})


def successful(value):
    return value["started"] and value["reaped"] and value["returncode"] == 0 and not any(
        value[k] for k in ("launch_failed","timed_out","interrupted"))


def reduce_events(plan, events):
    state = {"diagnosis":[],"generation":[],"workspace":None,"terminal":None,"blocked":None}
    for event in events:
        kind, data = event["kind"], event["data"]
        require(state["terminal"] is None and state["blocked"] is None)
        if kind == "infrastructure_failed":
            io._keys(data,("code",))
            require(data["code"] in ("PREPARATION_FAILED","MODEL_FAILED","CONTROLLER_IO","VALIDATION_IO","READINESS_ABORTED"))
            state["blocked"] = data
        elif kind == "workspace_ready":
            io._keys(data,("artifact",))
            require(state["workspace"] is None and state["generation"] and "marker" not in state["generation"][-1])
            state["workspace"] = data["artifact"]
        elif kind == "operation_terminal":
            io._keys(data,("outcome","artifact","candidate_digest"))
            require(data["outcome"] in ("candidate","candidate_invalid","diagnosis_invalid"))
            if data["outcome"] == "diagnosis_invalid":
                require(len(state["diagnosis"]) == plan.identity["diagnosis"]["max_attempts"]
                        and state["diagnosis"][-1].get("result",{}).get("outcome") == "invalid")
            else:
                require(state["generation"] and "terminal" in state["generation"][-1]
                        and successful(state["generation"][-1]["terminal"]["evidence"]))
            io._sha(data["candidate_digest"])
            # Consumption binds the immutable terminal content. The event hash
            # still verifies the entire physical history, including receipts,
            # but runtime wall times/authority digests cannot seed future plans.
            terminal_digest = io.digest({"schema_version": SCHEMA, "operation_id": plan.operation_id, **data})
            state["terminal"] = {**data, "terminal_digest": terminal_digest}
        else:
            stage, separator, action = kind.partition("_")
            require(separator and stage in ("diagnosis","generation"))
            attempts = state[stage]
            if action == "reserved":
                io._keys(data,("attempt","prompt_digest"))
                require(type(data["attempt"]) is int and data["attempt"] == len(attempts)+1)
                io._sha(data["prompt_digest"])
                if stage == "diagnosis":
                    require(not state["generation"] and len(attempts)<plan.identity["diagnosis"]["max_attempts"])
                    require(not attempts or attempts[-1].get("result",{}).get("outcome") == "invalid")
                    require(data["prompt_digest"] == plan.identity["diagnosis"]["prompt_digest"])
                else:
                    require(not attempts and state["diagnosis"] and state["diagnosis"][-1].get("result",{}).get("outcome") == "valid")
                attempts.append(dict(data))
                continue
            if action == "abandoned":
                # Operator-authorized abandonment of a failed, side-effect-free
                # (read-only) diagnosis attempt: it counts as an invalid attempt,
                # so the existing retry budget and diagnosis_invalid rule apply.
                io._keys(data,("attempt","artifact"))
                require(stage == "diagnosis" and attempts and "result" not in attempts[-1]
                        and "terminal" in attempts[-1] and not successful(attempts[-1]["terminal"]["evidence"])
                        and data["attempt"] == attempts[-1]["attempt"])
                attempts[-1]["result"] = {"attempt": data["attempt"], "outcome": "invalid",
                                          "artifact": data["artifact"], "abandoned": True}
                continue
            require(attempts and "result" not in attempts[-1])
            attempt = attempts[-1]
            require(data.get("attempt") == attempt["attempt"])
            if action == "marker":
                io._keys(data,("attempt","call_id","receipt"))
                require("marker" not in attempt)
                require(data["call_id"] == call_id(plan,stage,attempt["attempt"]),"BLOCK_IDENTITY_MISMATCH")
                require(stage != "generation" or state["workspace"] is not None)
                if data["receipt"] is not None:
                    io._keys(data["receipt"],("schema_version","config_digest","success_time","monotonic_time",
                                              "valid_for_seconds","category","receipt_digest"))
                    receipt = data["receipt"]
                    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
                            and receipt["category"] == "HTTPS_RESPONSE")
                    for key in ("config_digest", "receipt_digest"): io._sha(receipt[key])
                    for key in ("success_time", "monotonic_time", "valid_for_seconds"):
                        require(type(receipt[key]) in (int, float) and receipt[key] >= 0)
                    require(receipt["valid_for_seconds"] > 0)
                attempt["marker"] = data
            elif action == "started":
                io._keys(data,("attempt",))
                require("marker" in attempt and "started" not in attempt and "terminal" not in attempt)
                attempt["started"] = True
            elif action == "terminal":
                io._keys(data,("attempt","evidence","artifact"))
                require("marker" in attempt and "terminal" not in attempt)
                _terminal(data["evidence"])
                require(data["evidence"]["started"] == attempt.get("started",False))
                attempt["terminal"] = data
            elif action == "result":
                io._keys(data,("attempt","outcome","artifact"))
                require(stage == "diagnosis" and "terminal" in attempt and successful(attempt["terminal"]["evidence"]))
                require(data["outcome"] in ("valid","invalid") and data["artifact"]==attempt["terminal"]["artifact"])
                attempt["result"] = data
            else:
                require(False)
    return state


def classify(state, consumed=None):
    if state is None:
        return "OPERATION_UNPLANNED"
    if consumed is not None:
        terminal = state["terminal"]
        require(terminal is not None and consumed == {
            "operation_terminal_digest": terminal["terminal_digest"], "candidate_digest": terminal["candidate_digest"]},
            "BLOCK_IDENTITY_MISMATCH")
        return "OPERATION_COMMITTED"
    if state["terminal"]:
        return "COMMIT_CANDIDATE" if state["terminal"]["outcome"]=="candidate" else "COMMIT_CANDIDATE_INVALID"
    if state["blocked"]: return "BLOCK_INFRASTRUCTURE_FAILURE"
    for stage in ("generation","diagnosis"):
        if not state[stage]: continue
        current=state[stage][-1]
        if "terminal" in current and "result" not in current and not successful(current["terminal"]["evidence"]):
            return "BLOCK_INFRASTRUCTURE_FAILURE"
        if "marker" in current and "terminal" not in current:
            return "BLOCK_AMBIGUOUS_"+stage.upper()
        if "terminal" in current and "result" not in current:
            return "RECONCILE_"+stage.upper()+"_RESULT"
        if stage=="generation":return "START_GENERATION"
        if current.get("result",{}).get("outcome")=="valid":return "START_GENERATION"
        if current.get("result",{}).get("outcome")=="invalid":return "RESUME_DIAGNOSIS_RETRY"
        return "START_DIAGNOSIS"
    return "OPERATION_PLANNED"


class SearchOperationJournal:
    def __init__(self, directory, run_id, fingerprint, *, fault=lambda point:None):
        self.directory=io._checked_path(Path(directory)/"search_operations")
        self.run_id,self.fingerprint,self.fault=run_id,fingerprint,fault
        self._sessions = {}

    def validate(self, plan):
        require(plan.identity["run_id"]==self.run_id and plan.identity["run_fingerprint"]==self.fingerprint,
                "BLOCK_IDENTITY_MISMATCH")

    def folder(self, plan):
        self.validate(plan)
        return io._checked_path(self.directory/"operations"/plan.operation_id)

    def plans(self):
        if not self.directory.exists():return []
        io._checked_path(self.directory)
        path=self.directory/"head.json"
        if not path.exists():return []
        head=io._read_json(path);io._keys(head,("schema_version","operations","sha256"))
        require(type(head["schema_version"]) is int and head["schema_version"]==SCHEMA
                and type(head["operations"]) is list and head["sha256"]==io.digest(head["operations"]))
        plans=[]
        for ordinal, record in enumerate(head["operations"]):
            io._keys(record,("operation_id","plan_digest"));io._id(record["operation_id"])
            plan=SearchOperationPlan.from_dict(io._read_json(self.directory/"operations"/record["operation_id"]/"plan.json"))
            self.validate(plan)
            require(plan.identity["ordinal"]==ordinal and plan.digest==record["plan_digest"]
                    and plan.operation_id == record["operation_id"], "BLOCK_IDENTITY_MISMATCH")
            plans.append(plan)
        return plans

    def publish(self, plan):
        self.validate(plan)
        io._mkdir(self.directory)
        with io._exclusive(self.directory/".lock"):
            plans=self.plans();ordinal=plan.identity["ordinal"]
            if ordinal<len(plans):
                require(plans[ordinal]==plan,"BLOCK_IDENTITY_MISMATCH");return
            require(ordinal==len(plans),"BLOCK_IDENTITY_MISMATCH")
            io._mkdir(self.directory/"operations");folder=self.folder(plan);io._mkdir(folder)
            file=folder/"plan.json"
            if file.exists():require(io.same_json(io._read_json(file),plan.to_dict()),"BLOCK_IDENTITY_MISMATCH")
            else:durable_record(file,plan.to_dict(),fault=lambda p:self.fault("plan."+p))
            io._sync_dir(folder)  # Also re-establish a prior rename-before-fsync plan.
            for sub in ("events","artifacts","workspace"):io._mkdir(folder/sub)
            records=[{"operation_id":p.operation_id,"plan_digest":p.digest} for p in [*plans,plan]]
            durable_json(self.directory/"head.json",{"schema_version":SCHEMA,"operations":records,
                "sha256":io.digest(records)},replace_existing=True,fault=lambda p:self.fault("catalog."+p))

    def inspect(self, plan):
        folder=self.folder(plan)
        require(io.same_json(io._read_json(folder/"plan.json"),plan.to_dict()),"BLOCK_IDENTITY_MISMATCH")
        io._checked_path(folder/"events")
        paths=sorted(p for p in (folder/"events").iterdir() if not p.name.startswith(".tmp-"))
        events=[];previous=plan.digest
        for seq,path in enumerate(paths,1):
            require(path.name==f"{seq:012d}.json")
            row=io._read_json(path)
            io._keys(row,("schema_version","operation_id","sequence","previous_digest","kind","data","sha256"))
            require(type(row["schema_version"]) is int and row["schema_version"]==SCHEMA
                and type(row["sequence"]) is int and row["sequence"]==seq
                and row["operation_id"]==plan.operation_id and row["previous_digest"]==previous)
            require(row["sha256"]==io.digest({k:v for k,v in row.items() if k!="sha256"}))
            previous=row["sha256"];events.append(row)
        head=folder/"durable_head.json"
        if head.exists():
            value=io._read_json(head);io._keys(value,("schema_version","sequence","sha256"))
            n=value["sequence"]
            require(type(value["schema_version"]) is int and value["schema_version"]==SCHEMA and type(n) is int and 1<=n<=len(events)
                    and events[n-1]["sha256"]==value["sha256"])
        state=reduce_events(plan,events)
        for row in events:
            self.validate_event_artifact(plan, row)
        return state,events

    def validate_event_artifact(self, plan, row):
        artifact = row["data"].get("artifact")
        if artifact is None:
            return
        value = self.read_artifact(plan, artifact)
        if row["kind"] == "operation_terminal":
            require(value["kind"] == row["data"]["outcome"])
            expected = value["candidate_digest"] if value["kind"] == "candidate" else artifact["sha256"]
            require(row["data"]["candidate_digest"] == expected, "BLOCK_IDENTITY_MISMATCH")
        elif row["kind"] in ("diagnosis_terminal", "generation_terminal"):
            expected = (row["kind"].split("_")[0] + "_result" if
                        successful(row["data"]["evidence"]) else "process_failure")
            require(value["kind"] == expected)
        elif row["kind"] == "diagnosis_result":
            require(value["kind"] == "diagnosis_result" and row["data"]["outcome"] == value["outcome"])
        elif row["kind"] == "diagnosis_abandoned":
            require(value["kind"] == "diagnosis_abandoned" and value["attempt"] == row["data"]["attempt"])
        elif row["kind"] == "workspace_ready":
            require(value["kind"] == "workspace")

    def read_artifact(self, plan, reference):
        io._keys(reference,("name","sha256"));io._id(reference["name"]);io._sha(reference["sha256"])
        value=io._read_json(self.folder(plan)/"artifacts"/reference["name"])
        require(io.digest(value)==reference["sha256"])
        require(value.get("operation_id")==plan.operation_id,"BLOCK_IDENTITY_MISMATCH")
        require(type(value.get("schema_version")) is int and value["schema_version"] == SCHEMA)
        kinds = {"diagnosis_result": ("outcome", "fields", "result_digest"),
                 "process_failure": (), "generation_result": ("valid_workspace", "source_digest"),
                 "workspace": ("baseline_tree", "parent_source", "git_config", "git_head", "git_management"),
                 "candidate": ("metadata", "patch_digest", "candidate_digest"),
                 "candidate_invalid": ("code",), "diagnosis_invalid": (),
                 "diagnosis_abandoned": ("attempt", "authorization")}
        require(value.get("kind") in kinds)
        io._keys(value, ("schema_version", "operation_id", "kind", *kinds[value["kind"]]))
        if value["kind"] == "diagnosis_result":
            require(value["outcome"] in ("valid", "invalid") and value["result_digest"] == io.digest(value["fields"]))
            if value["outcome"] == "invalid":
                require(value["fields"] is None)
            else:
                from search_operations import diagnosis_result
                require(diagnosis_result(io.canonical(value["fields"]))["outcome"] == "valid")
        elif value["kind"] == "generation_result":
            require(type(value["valid_workspace"]) is bool)
            if value["valid_workspace"]: io._sha(value["source_digest"])
            else: require(value["source_digest"] is None)
        elif value["kind"] == "workspace":
            require(isinstance(value["baseline_tree"], str) and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", value["baseline_tree"]))
            for key in ("parent_source", "git_config", "git_head", "git_management"): io._sha(value[key])
        elif value["kind"] == "candidate":
            for key in ("patch_digest", "candidate_digest"): io._sha(value[key])
        elif value["kind"] == "candidate_invalid":
            require(value["code"] in ("WORKSPACE_INVALID", "TESTS_FAILED", "PATCH_TOO_LARGE", "PATCH_INVALID"))
        return value

    @contextmanager
    def session(self,plan):
        folder=self.folder(plan)
        with io._exclusive(folder/".lock"):
            session=Session(self,plan)
            self._sessions[plan.operation_id] = session
            try:yield session
            finally:
                session.active=False
                self._sessions.pop(plan.operation_id, None)


class Session:
    def __init__(self,journal,plan):
        self.journal,self.plan=journal,plan
        self.active=True;self.poisoned=False
        self.pid = os.getpid()
        self._staging = {}
        self.state,self.events=journal.inspect(plan)
        # Re-establish publication durability under the same operation lease.
        io._sync_dir(journal.folder(plan)/"events")

    def check(self):
        require(self.active and not self.poisoned and self.pid == os.getpid()
                and self.journal._sessions.get(self.plan.operation_id) is self, "OPERATION_SESSION_CLOSED")

    def refresh(self):
        self.check()
        state,events=self.journal.inspect(self.plan)
        require(events==self.events,"BLOCK_IDENTITY_MISMATCH")
        return state

    def diagnosis_staging(self, attempt):
        self.check()
        require(attempt not in self._staging, "DIAGNOSIS_STAGING_INVALID")
        owner = DiagnosisStaging(self, attempt)
        self._staging[attempt] = owner
        return owner

    def artifact(self,name,value):
        self.check();io._id(name)
        value={"schema_version":SCHEMA,"operation_id":self.plan.operation_id,**value}
        file=self.journal.folder(self.plan)/"artifacts"/name
        if file.exists():require(io.same_json(io._read_json(file),value),"BLOCK_IDENTITY_MISMATCH")
        else:
            try: durable_record(file,value,fault=lambda p:self.journal.fault("artifact."+p))
            except BaseException:
                self.poisoned=True
                raise
        return {"name":name,"sha256":io.digest(value)}

    def append(self,kind,data):
        self.check();self.refresh()
        body={"schema_version":SCHEMA,"operation_id":self.plan.operation_id,"sequence":len(self.events)+1,
              "previous_digest":self.events[-1]["sha256"] if self.events else self.plan.digest,"kind":kind,"data":data}
        row={**body,"sha256":io.digest(body)}
        reduce_events(self.plan,[*self.events,row])
        self.journal.validate_event_artifact(self.plan, row)
        folder=self.journal.folder(self.plan)
        try:
            durable_record(folder/"events"/f"{row['sequence']:012d}.json",row,
                         fault=lambda p:self.journal.fault(kind+"."+p))
            durable_json(folder/"durable_head.json",{"schema_version":SCHEMA,"sequence":row["sequence"],"sha256":row["sha256"]},
                         replace_existing=True,fault=lambda p:self.journal.fault(kind+".head."+p))
        except BaseException:
            self.poisoned=True;raise
        self.events.append(row);self.state=reduce_events(self.plan,self.events)
        return row


class DiagnosisStaging:
    """In-memory lease for one operation/attempt; never a serialized authority."""
    LIMIT = 256 * 1024

    def __init__(self, session, attempt):
        self.session, self.attempt = session, attempt
        self.path = session.journal.folder(session.plan) / "workspace" / f"diagnosis-attempt-{attempt:06d}.staging"
        self.call_id = call_id(session.plan, "diagnosis", attempt)
        self._raw = None
        self._inode = None
        self._used = False

    def check(self, path, identity):
        self.session.check()
        require(self.session._staging.get(self.attempt) is self, "DIAGNOSIS_STAGING_INVALID")
        expected = (self.session.journal.folder(self.session.plan) / "workspace" /
                    f"diagnosis-attempt-{self.attempt:06d}.staging")
        require(io._checked_path(path) == self.path == expected and Path(path).is_absolute()
                and identity == self.call_id == call_id(self.session.plan, "diagnosis", self.attempt),
                "DIAGNOSIS_STAGING_INVALID")
        attempt = self.session.state["diagnosis"][-1]
        require(attempt["attempt"] == self.attempt and "terminal" not in attempt
                and attempt.get("marker", {}).get("call_id") == identity, "DIAGNOSIS_STAGING_INVALID")

    def start(self, path, identity):
        self.check(path, identity)
        require(not self._used and self.path.parent.is_dir() and not self.path.exists(),
                "DIAGNOSIS_STAGING_INVALID")
        self._used = True

    def read(self):
        self.check(self.path, self.call_id)
        if self._raw is not None:
            self._check_inode()
            return self._raw
        # O_NONBLOCK also prevents a raced FIFO from blocking the controller.
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and info.st_size <= self.LIMIT, "DIAGNOSIS_STAGING_INVALID")
            self._inode = (info.st_dev, info.st_ino)
            self._check_inode()
            raw = stream.read(self.LIMIT + 1)
            require(len(raw) <= self.LIMIT and len(raw) == info.st_size, "DIAGNOSIS_STAGING_INVALID")
            os.fsync(stream.fileno())
            self._check_inode()
        io._sync_dir(self.path.parent)
        self._raw = raw
        return raw

    def _check_inode(self):
        info = io._regular(self.path).lstat()
        require((info.st_dev, info.st_ino) == self._inode, "DIAGNOSIS_STAGING_INVALID")

    def cleanup(self):
        self.check(self.path, self.call_id)
        if self._inode is not None:
            self._check_inode()
        elif not self.path.exists():
            return
        else:
            io._regular(self.path)
        self.path.unlink()
        io._sync_dir(self.path.parent)
