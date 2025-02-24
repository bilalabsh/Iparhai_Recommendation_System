const { exec } = require("child_process");
const path = require("path");

export default function handler(req, res) {
  // Construct the path to the Python script
  const pythonScript = path.join(__dirname, "../backend/main2.py");

  // Execute the Python script
  exec(`python3 ${pythonScript}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error.message}`);
      res.status(500).json({ error: error.message });
      return;
    }
    if (stderr) {
      console.error(`Stderr: ${stderr}`);
      res.status(500).json({ error: stderr });
      return;
    }

    console.log(`Output: ${stdout}`);
    res.status(200).json({ message: stdout });
  });
}
