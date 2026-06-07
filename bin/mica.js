#!/usr/bin/env node
"use strict";

const { spawnSync, spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const packageRoot = path.resolve(__dirname, "..");
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"));

const BANNER = String.raw`
 __  __  ___   ____    _    
|  \/  ||_ _| / ___|  / \   
| |\/| | | | | |     / _ \  
| |  | | | | | |___ / ___ \ 
|_|  |_||___| \____/_/   \_\
`;

function printBanner() {
  process.stdout.write(`${BANNER}\n`);
  process.stdout.write("Molecular Inference of Compatibility and Affinity\n");
  process.stdout.write(`MICA ${packageJson.version} | high-throughput solubility inference\n\n`);
}

function pythonCandidates() {
  const candidates = [];
  if (process.env.MICA_PYTHON) candidates.push(process.env.MICA_PYTHON);
  candidates.push("/usr/bin/python3", "python3", "python");
  return [...new Set(candidates)];
}

function testPython(candidate) {
  const result = spawnSync(candidate, ["-c", "import sys; print(sys.executable)"], {
    encoding: "utf8",
  });
  if (result.status !== 0) return null;
  return result.stdout.trim();
}

function findPython() {
  for (const candidate of pythonCandidates()) {
    const executable = testPython(candidate);
    if (executable) return { command: candidate, executable };
  }
  return null;
}

function envWithPythonPath() {
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${packageRoot}${path.delimiter}${env.PYTHONPATH}` : packageRoot;
  return env;
}

function runPython(args) {
  const python = findPython();
  if (!python) {
    console.error("MICA error: no Python interpreter found. Set MICA_PYTHON=/path/to/python.");
    process.exit(1);
  }
  const child = spawn(python.command, ["-m", "mica", ...args], {
    cwd: packageRoot,
    env: envWithPythonPath(),
    stdio: "inherit",
  });
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    process.exit(code ?? 1);
  });
}

function doctor() {
  printBanner();
  const python = findPython();
  if (!python) {
    console.log("python: FAIL");
    console.log("  No usable Python interpreter found.");
    console.log("  Set MICA_PYTHON=/path/to/python and retry.");
    process.exit(1);
  }

  console.log(`node:   ${process.version}`);
  console.log(`npm pkg: ${packageJson.name}@${packageJson.version}`);
  console.log(`root:   ${packageRoot}`);
  console.log(`python: ${python.executable}`);

  const check = [
    "import importlib.util as u",
    "mods=['numpy','pandas','sklearn','xgboost']",
    "missing=[m for m in mods if u.find_spec(m) is None]",
    "print('missing=' + ','.join(missing))",
    "raise SystemExit(1 if missing else 0)",
  ].join("; ");
  const result = spawnSync(python.command, ["-c", check], {
    encoding: "utf8",
    env: envWithPythonPath(),
  });
  const missingLine = (result.stdout || "").trim();
  if (result.status === 0) {
    console.log("python deps: PASS");
    console.log("\nMICA is ready.");
    return;
  }
  const missing = missingLine.replace(/^missing=/, "");
  console.log("python deps: FAIL");
  console.log(`missing: ${missing || "unknown"}`);
  console.log("\nInstall dependencies in the selected Python environment:");
  console.log("  python -m pip install numpy pandas scikit-learn xgboost");
  process.exit(1);
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    printBanner();
    runPython(["--help"]);
    return;
  }
  if (args.includes("--version") || args.includes("-V")) {
    console.log(`MICA ${packageJson.version}`);
    return;
  }
  if (args[0] === "doctor") {
    doctor();
    return;
  }
  if (args[0] === "--help" || args[0] === "-h") {
    printBanner();
    runPython(["--help"]);
    return;
  }
  runPython(args);
}

main();

