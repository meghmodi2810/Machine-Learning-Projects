// static/app.js

function sendCommand(command) {
  fetch(`/command/${command}`, { method: 'POST' })
    .then(res => res.text())
    .then(msg => {
      document.getElementById('statusMsg').innerText = 'Status: ' + msg;
    })
    .catch(err => {
      console.error('Command error:', err);
      document.getElementById('statusMsg').innerText = 'Error sending command';
    });
}

function updateSensitivity(value) {
  fetch('/set_sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value: parseInt(value) })
  })
    .then(res => res.text())
    .then(msg => {
      document.getElementById('statusMsg').innerText = 'Sensitivity: ' + value;
    });
}

function togglePrivacyBlur(enabled) {
  fetch('/toggle_privacy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blur: enabled })
  })
    .then(res => res.text())
    .then(msg => {
      document.getElementById('statusMsg').innerText = 'Privacy Blur: ' + (enabled ? 'ON' : 'OFF');
    });
}

function uploadVideo(file) {
  const formData = new FormData();
  formData.append('video', file);

  fetch('/upload_video', {
    method: 'POST',
    body: formData
  })
    .then(res => res.text())
    .then(msg => {
      document.getElementById('statusMsg').innerText = 'Upload: ' + msg;
    });
}
