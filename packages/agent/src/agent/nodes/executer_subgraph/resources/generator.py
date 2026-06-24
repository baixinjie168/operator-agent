"""ATK API Script Generator.

Generates an ATK-compatible .py executor file from test case JSON + signature table.

Usage:
    python generator.py <cases.json> [-o output.py] --signatures aclnn_extracted.txt
"""

import argparse
import json
import sys
from pathlib import Path


def load_signatures(sig_path: str) -> dict[str, str]:
    sig_map: dict[str, str] = {}
    p = Path(sig_path)
    if not p.exists():
        print(f"Warning: signature file not found: {sig_path}", file=sys.stderr)
        return sig_map
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) >= 2:
            api_name = parts[0].strip()
            signature = parts[1].strip()
            sig_map[api_name] = signature
    return sig_map


def generate_executor(cases: list[dict], sig_map: dict[str, str], operator_name: str) -> str:
    aclnn_name = cases[0].get("aclnn_name", operator_name) if cases else operator_name
    aclnn_api = f"aclnn{aclnn_name}"
    signature = sig_map.get(aclnn_api, f"aclnn{aclnn_name}(x, weight, bias, ...)")

    lines = [
        f'"""ATK Executor for {operator_name}."""',
        "import torch",
        "import torch_npu",
        "from atk import ATKTestCase, ATKRunner",
        "",
        "",
        f"class {aclnn_name}Executor(ATKTestCase):",
        f'    """Auto-generated ATK executor for {aclnn_api}."""',
        "",
        f"    aclnn_api = '{aclnn_api}'",
        f"    aclnn_signature = '{signature}'",
        "",
        "    def setup(self):",
        "        self.inputs = self.case_data.get('inputs', [])",
        "        self.outputs = self.case_data.get('outputs', [])",
        "",
        "    def npu_execute(self):",
        f"        return torch.ops.aten.{aclnn_name.lower()}.default(*self._build_args())",
        "",
        "    def _build_args(self):",
        "        args = []",
        "        for inp in self.inputs:",
        "            if inp.get('type') == 'tensor':",
        "                t = torch.randn(inp.get('shape', [2, 2]),",
        "                               dtype=self._map_dtype(inp.get('dtype', 'float32')))",
        "                args.append(t)",
        "            else:",
        "                args.append(inp.get('value', 0))",
        "        return args",
        "",
        "    @staticmethod",
        "    def _map_dtype(dtype_str: str):",
        "        mapping = {",
        "            'float32': torch.float32,",
        "            'float16': torch.float16,",
        "            'bfloat16': torch.bfloat16,",
        "            'int32': torch.int32,",
        "            'int64': torch.int64,",
        "        }",
        "        return mapping.get(dtype_str, torch.float32)",
        "",
        "",
        "def cpu_golden_reference(*args, **kwargs):",
        '    """CPU golden reference — to be filled by CPU derivation step."""',
        "    raise NotImplementedError('CPU golden reference not yet implemented')",
        "",
        "",
        "if __name__ == '__main__':",
        f"    ATKRunner('{operator_name}', {aclnn_name}Executor).run()",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ATK API Script Generator")
    parser.add_argument("cases_json", help="Path to test cases JSON file")
    parser.add_argument("-o", "--output", help="Output .py file path")
    parser.add_argument("--signatures", required=True, help="Path to aclnn_extracted.txt")
    args = parser.parse_args()

    cases_path = Path(args.cases_json)
    if not cases_path.exists():
        print(f"Error: cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(1)

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    operator_name = cases[0].get("name", cases_path.stem.replace("_cases", "")) if cases else "Unknown"

    sig_map = load_signatures(args.signatures)

    output_path = args.output or f"{operator_name}_atk_executor.py"
    executor_code = generate_executor(cases, sig_map, operator_name)
    Path(output_path).write_text(executor_code, encoding="utf-8")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
