import datasets
from typing import List
import base64, zlib, pickle, json, ast, tempfile, subprocess, importlib.util, sys
from pathlib import Path
import io, runpy, signal, sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Write a **complete** Python 3 "
    "solution that {}. Include no explanation and no comments.\n\n"
    "# Problem:\n"
)
TIMEOUT = 12

STDIN_PROMPT = "reads from input and prints the answer"
FUNCTIONAL_PROMPT = (
    "completes the required function so that entire code can be run as a script"
)

current_dir = Path(__file__).parent


def run_stdin_slower(path: Path, stdin: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, path],
            input=stdin.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=TIMEOUT,
        )
        return proc.stdout.decode()
    except Exception as e:
        return ""


def b64_zlib_pickle_to_json(b64: str) -> List[dict]:
    buf = zlib.decompress(base64.b64decode(b64))
    jtext = pickle.loads(buf)
    return json.loads(jtext)


def create_stdin_file(code: str) -> str:
    path = Path(current_dir, "main.py")
    path.write_text(code)
    return path


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        doc_to_return = {
            **doc,
            "public_test_cases": json.loads(doc["public_test_cases"]),
            "private_test_cases": b64_zlib_pickle_to_json(doc["private_test_cases"]),
            "func_name": json.loads(doc["metadata"] or "{}").get("func_name"),
        }

        return doc_to_return

    return dataset.map(_process_doc)


def doc_to_text(doc: dict):
    bool_stdin = doc["private_test_cases"][0]["testtype"] == "stdin"
    prompt = SYSTEM_PROMPT.format(STDIN_PROMPT if bool_stdin else FUNCTIONAL_PROMPT)
    prompt = prompt + doc["question_content"] + "\n```python\n" + doc["starter_code"]
    return prompt


def doc_to_target(doc: dict):
    bool_stdin = doc["private_test_cases"][0]["testtype"] == "stdin"
    test_cases_outputs = [
        (
            test_case["output"].rstrip()
            if bool_stdin
            else parse_literal(test_case["output"].strip())
        )
        for test_case in doc["public_test_cases"] + doc["private_test_cases"]
    ]
    return test_cases_outputs


def import_module_from_code(code: str):
    """Exec user's code once and return the loaded module object."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td, "user_mod.py")
        path.write_text(code)
        name = f"user_mod_{hash(code)}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # needed for relative imports
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            return None


def parse_literal(text: str):
    """JSON if possible, else Python literal-eval, else raw str."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text.strip()


def extract_args(arg_blob: str):
    """
    Convert the `input` field of a *functional* test-case into a tuple
    of positional arguments:

    • If it contains >=1 newline we split on lines, strip blanks,
      parse each chunk separately, and return *all* of them.
    • Otherwise we parse the whole blob once and return a 1-tuple.
    """
    lines = [ln.strip() for ln in arg_blob.strip().splitlines() if ln.strip()]
    if len(lines) > 1:
        return tuple(parse_literal(ln) for ln in lines)
    return (parse_literal(arg_blob),)


def create_exec_function(code: str, func_name: str):
    mod = import_module_from_code(code)
    if mod is None:
        return None
    # locate callable
    func = getattr(mod, func_name, None)
    if func is not None:
        return func
    if func is None and hasattr(mod, "Solution"):
        sol = mod.Solution()
        func = getattr(sol, func_name, None)
        return func
    if func is None or not callable(func):
        return None
    return None



class TimeoutExpired(Exception):
    pass

def _alarm_handler(signum, frame):
        raise TimeoutExpired(f"TIME LIMIT ERROR")

def run_functional(func, arg_str: str):
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    error_received = False
    result = None
    try:
        result = func(*extract_args(arg_str))
    except Exception as e:
        error_received = True
    finally:
        # always restore state, cancel timer
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    return result if not error_received else None
    



def run_stdin(path: Path, stdin_data: str) -> str:
    """
    Run the file at *path* (with runpy.run_path) while:
      • replacing sys.stdin with `stdin_data`
      • capturing sys.stdout
      • raising TimeoutExpired if it runs longer than `timeout` seconds
    Returns whatever the program printed.
    """

    with open(str(path), "r") as f:
        f_content = f.read()
        if "import threading" in f_content:
            return "None"


    fake_in, fake_out = io.StringIO(stdin_data), io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    error_received = False
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        sys.stdin, sys.stdout = fake_in, fake_out
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Exception as e:
            error_received = True
    finally:
        # always restore state, cancel timer
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        sys.stdin, sys.stdout = old_in, old_out

    return fake_out.getvalue() if not error_received else "None"


def pass_at_k(references, predictions, k: list[int] = None):
    return references[0] == predictions[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    code = "from typing import *\n"
    predictions = []
    for resp, doc in zip(resps, docs):

        _resp = resp[0]
        if "```" in _resp:
            _resp = _resp[:_resp.index("```")]


        generated_outputs = []
        adjusted_code = code + doc["starter_code"] + _resp
        bool_stdin = doc["private_test_cases"][0]["testtype"] == "stdin"

        stdin_file_path = create_stdin_file(adjusted_code) if bool_stdin else None
        executable_function = (
            None
            if bool_stdin
            else create_exec_function(adjusted_code, doc["func_name"])
        )
        if not bool_stdin and executable_function is None:
            generated_outputs.append("None")
            predictions.append(generated_outputs)
            continue

        for question in doc["public_test_cases"] + doc["private_test_cases"]:

            if question["testtype"] == "stdin":
                generated_module_output = run_stdin(
                    stdin_file_path, question["input"]
                ).rstrip()
                generated_outputs.append(generated_module_output)

            else:
                generated_module_output = run_functional(
                    executable_function, question["input"]
                )
                generated_outputs.append(generated_module_output)

        predictions.append(generated_outputs)
    return predictions
