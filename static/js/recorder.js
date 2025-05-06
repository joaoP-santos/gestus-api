let mediaRecorder;
let recordedChunks = [];
let isRecording = false;
let stream = null;
let videoPlayer = null;
let uploadedVideoFile = null;
let currentCameraId = null;

// Get available cameras and populate the select dropdown
async function enumerateCameras() {
  try {
    const cameraSelect = document.getElementById("cameraSelect");

    // Clear existing options except the first one
    while (cameraSelect.options.length > 1) {
      cameraSelect.remove(1);
    }
    cameraSelect.options[0].text = "Loading cameras...";

    // Get list of video input devices
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(
      (device) => device.kind === "videoinput"
    );

    if (videoDevices.length === 0) {
      cameraSelect.options[0].text = "No cameras found";
      return;
    }

    // Remove the loading option
    cameraSelect.remove(0);

    // Add camera options
    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      // Use device label if available, otherwise use "Camera X"
      option.text = device.label || `Camera ${index + 1}`;
      cameraSelect.add(option);

      // Select the first device by default if none is selected yet
      if (index === 0 && !currentCameraId) {
        currentCameraId = device.deviceId;
      }
    });

    // Select the previous camera if it's still available
    if (currentCameraId) {
      for (let i = 0; i < cameraSelect.options.length; i++) {
        if (cameraSelect.options[i].value === currentCameraId) {
          cameraSelect.selectedIndex = i;
          break;
        }
      }
    }

    console.log(`Found ${videoDevices.length} cameras`);
  } catch (err) {
    console.error("Error enumerating cameras:", err);
    document.getElementById("cameraSelect").options[0].text =
      "Error loading cameras";
  }
}

// Handle camera selection change
async function switchCamera() {
  const cameraSelect = document.getElementById("cameraSelect");
  const selectedCameraId = cameraSelect.value;

  if (!selectedCameraId || selectedCameraId === currentCameraId) {
    return;
  }

  // Update current camera ID
  currentCameraId = selectedCameraId;

  // Stop the previous stream if it exists
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  // Update status
  document.getElementById("status").innerText = "Switching camera...";

  // Reinitialize with the new camera
  await initializeCamera();

  document.getElementById("status").innerText = "Ready";
}

async function initializeCamera() {
  try {
    // If no cameras are selected yet, enumerate them first
    if (!currentCameraId) {
      await enumerateCameras();
      const cameraSelect = document.getElementById("cameraSelect");
      if (cameraSelect.options.length > 0) {
        currentCameraId = cameraSelect.value;
      }
    }

    // Get access to the camera
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: currentCameraId ? { exact: currentCameraId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    videoPlayer = document.getElementById("videoPlayer");
    videoPlayer.srcObject = stream;

    // If this is the first time, populate the camera list
    if (!document.getElementById("cameraSelect").options.length) {
      await enumerateCameras();
    }
  } catch (err) {
    console.error("Error accessing webcam:", err);
    document.getElementById("status").innerText = "Camera error";
    alert(
      "Error accessing webcam. Please ensure you have a webcam connected and have granted permission."
    );
  }
}

// Setup event listeners for camera controls
function setupCameraControls() {
  const cameraSelect = document.getElementById("cameraSelect");
  const refreshBtn = document.getElementById("refreshCamerasBtn");

  // Handle camera selection change
  cameraSelect.addEventListener("change", switchCamera);

  // Handle camera refresh button click
  refreshBtn.addEventListener("click", async () => {
    // Update button to show it's working
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = '<span style="font-size: 14px;">⌛</span>';

    // Re-enumerate cameras
    await enumerateCameras();

    // Switch to the selected camera
    await switchCamera();

    // Reset button
    refreshBtn.disabled = false;
    refreshBtn.innerHTML = '<span style="font-size: 14px;">⟳</span>';
  });
}

// Initialize camera controls when page loads
document.addEventListener("DOMContentLoaded", setupCameraControls);

function startRecording() {
  if (!stream) return;
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = uploadVideo;

  mediaRecorder.start();
  isRecording = true;

  // Make sure UI updates immediately
  updateUI();

  // Set recording status
  document.getElementById("status").innerText = "Recording...";

  // Explicitly set display properties for buttons
  document.getElementById("startBtn").style.display = "none";
  document.getElementById("stopBtn").style.display = "block";
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    updateUI();
  }
}

function updateUI() {
  document.getElementById("startBtn").style.display = isRecording
    ? "none"
    : "block";
  document.getElementById("stopBtn").style.display = isRecording
    ? "block"
    : "none";
  document.getElementById("status").innerText = isRecording
    ? "Recording..."
    : "Ready";
  document.getElementById("loading").style.display = "none";
}

async function uploadVideo() {
  const blob = new Blob(recordedChunks, { type: "video/webm" });

  // Check if the video is too short
  if (blob.size < 10000) {
    // Less than ~10KB is probably too short
    document.getElementById("status").innerText =
      "Recording too short, please record a longer video";
    return;
  }

  const formData = new FormData();
  formData.append("video", blob);

  try {
    document.getElementById("status").innerText = "Processing...";
    document.getElementById("loading").style.display = "flex";

    const response = await fetch("/process", {
      method: "POST",
      body: formData,
    });

    document.getElementById("loading").style.display = "none";

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `Server error: ${response.status}`);
    }

    displayResults(result);
  } catch (error) {
    console.error("Error processing video:", error);
    document.getElementById("status").innerText = `Error: ${
      error.message || "Could not process video"
    }`;
    document.getElementById("loading").style.display = "none";

    // Show helpful message
    document.getElementById("results").innerHTML =
      "<p>Please try again with clear hand movements and ensure both hands are visible.</p>";
  }
}

function displayResults(results) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (results.status === "error") {
    document.getElementById("status").innerText = "Error";
    resultsDiv.innerHTML = `<p class="error">${
      results.message || "An error occurred during processing"
    }</p>`;
    return;
  }

  if (results.predictions && results.predictions.length > 0) {
    document.getElementById("status").innerText = "Done!";

    const table = document.createElement("table");
    table.className = "results-table";

    // Add header
    const header = table.createTHead();
    const headerRow = header.insertRow();
    ["Sign", "Confidence"].forEach((text) => {
      const th = document.createElement("th");
      th.textContent = text;
      headerRow.appendChild(th);
    });

    // Add predictions
    const tbody = table.createTBody();
    results.predictions.forEach((pred) => {
      const row = tbody.insertRow();
      const signCell = row.insertCell();
      const confCell = row.insertCell();

      signCell.textContent = pred.sign;

      // Color confidence based on value
      const confidence = pred.confidence * 100;
      confCell.textContent = `${confidence.toFixed(1)}%`;

      if (confidence >= 80) {
        confCell.style.color = "#4CAF50"; // Green for high confidence
      } else if (confidence >= 50) {
        confCell.style.color = "#FF9800"; // Orange for medium confidence
      } else {
        confCell.style.color = "#f44336"; // Red for low confidence
      }
    });

    resultsDiv.appendChild(table);
  } else {
    document.getElementById("status").innerText = "No signs detected";
    resultsDiv.innerHTML =
      "<p>No signs could be recognized in this video. Try again with clearer hand movements.</p>";
  }
}

function switchTab(tabName) {
  // Hide all tab content
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.remove("active");
  });

  // Deactivate all tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.remove("active");
  });

  // Activate selected tab and content
  document.getElementById(`${tabName}-tab`).classList.add("active");
  document
    .querySelector(`.tab:nth-child(${tabName === "record" ? 1 : 2})`)
    .classList.add("active");

  if (tabName === "record") {
    // Make sure camera is initialized when switching to record tab
    initializeCamera();
  }

  // Reset status
  document.getElementById("status").innerText = "Ready";
  document.getElementById("results").innerHTML = "";
}

function handleFileSelect(event) {
  const fileInput = event.target;
  uploadedVideoFile = fileInput.files[0];

  if (uploadedVideoFile) {
    const fileSize = (uploadedVideoFile.size / (1024 * 1024)).toFixed(2); // Convert to MB

    // Update file info
    document.getElementById("file-info").innerHTML = `
      Selected: <strong>${uploadedVideoFile.name}</strong><br>
      Size: ${fileSize} MB | Type: ${uploadedVideoFile.type}
    `;

    // Enable upload button
    document.getElementById("uploadBtn").disabled = false;
  } else {
    document.getElementById("file-info").innerHTML = "";
    document.getElementById("uploadBtn").disabled = true;
  }
}

function processUploadedVideo() {
  if (!uploadedVideoFile) {
    document.getElementById("status").innerText =
      "Please select a video file first";
    return;
  }

  // Create form data
  const formData = new FormData();
  formData.append("video", uploadedVideoFile);

  // Process the upload
  document.getElementById("status").innerText = "Processing...";
  document.getElementById("loading").style.display = "flex";

  // Send to server for processing
  fetch("/process", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      document.getElementById("loading").style.display = "none";

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      return response.json();
    })
    .then((result) => {
      displayResults(result);
    })
    .catch((error) => {
      console.error("Error processing video:", error);
      document.getElementById("status").innerText = `Error: ${
        error.message || "Could not process video"
      }`;
      document.getElementById("loading").style.display = "none";

      document.getElementById("results").innerHTML =
        "<p>There was a problem processing your video. Please try again or use a different file.</p>";
    });
}

// Add drag and drop functionality for the upload area
document.addEventListener("DOMContentLoaded", function () {
  const dropArea = document.querySelector(".file-upload");

  // Prevent default drag behaviors
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  // Highlight drop area when item is dragged over
  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  // Handle dropped files
  dropArea.addEventListener("drop", handleDrop, false);

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight() {
    dropArea.style.borderColor = "#4caf50";
    dropArea.style.backgroundColor = "#f1f8e9";
  }

  function unhighlight() {
    dropArea.style.borderColor = "#ccc";
    dropArea.style.backgroundColor = "";
  }

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];

    if (file && file.type.startsWith("video/")) {
      document.getElementById("video-file").files = dt.files;
      handleFileSelect({ target: document.getElementById("video-file") });
    } else {
      document.getElementById("file-info").innerHTML =
        '<span style="color:red;">Please drop a video file</span>';
    }
  }
});

// Initialize when page loads
window.addEventListener("load", initializeCamera);
