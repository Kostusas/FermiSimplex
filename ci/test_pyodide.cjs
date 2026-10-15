// Run the installed-wheel numerical smoke tests in the actual Wasm runtime.
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const wheel = path.resolve(process.argv[2]);
  const runtime = path.resolve(process.argv[3]);
  const { loadPyodide } = require(runtime);
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(['numpy', 'scipy', 'micropip', 'pytest']);
  const filename = path.basename(wheel);
  pyodide.FS.writeFile(`/tmp/${filename}`, fs.readFileSync(wheel));
  pyodide.FS.writeFile('/tmp/test_wheel.py', fs.readFileSync(path.join(__dirname, 'test_wheel.py')));
  pyodide.globals.set('wheel_url', `emfs:/tmp/${filename}`);
  await pyodide.runPythonAsync('import micropip\nawait micropip.install(wheel_url)');
  const status = pyodide.runPython('import pytest\npytest.main(["/tmp/test_wheel.py", "-q"])');
  if (status !== 0) throw new Error(`Wasm wheel tests failed: ${status}`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
