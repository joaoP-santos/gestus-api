let mediaRecorder;
let recordedChunks = [];
let isRecording = false;
let stream = null;
let videoPlayer = null;
let currentCameraId = null;
let currentSign = null;
let userContributions = 0;

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

// Get signs data from server
async function fetchSignsData() {
  try {
    const response = await fetch("/api/dataset-stats");
    const data = await response.json();

    // Update the UI with the statistics
    document.getElementById("total-count").innerText = data.totalSamples;

    // Load from local storage or initialize to 0
    userContributions = parseInt(
      localStorage.getItem("userContributions") || "0"
    );
    document.getElementById("contributions-count").innerText =
      userContributions;

    return data;
  } catch (error) {
    console.error("Error fetching signs data:", error);
    return { signs: ["Error loading signs"] };
  }
}

// Get a random sign prompt
async function getNextSign() {
  try {
    const response = await fetch("/api/get-random-sign");
    const data = await response.json();
    currentSign = data.sign;

    // Update the UI
    document.getElementById("current-sign").innerText = currentSign;
    document.getElementById("status").innerText = "Ready to record";

    // Hide any previous success message
    document.getElementById("success-message").style.display = "none";
  } catch (error) {
    console.error("Error getting random sign:", error);
    document.getElementById("current-sign").innerText = "Error loading sign";
  }
}

function startRecording() {
  if (!stream || !currentSign) return;

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = uploadLandmarks;

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

async function uploadLandmarks() {
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
  formData.append("sign", currentSign);

  try {
    document.getElementById("status").innerText = "Processing...";
    document.getElementById("loading").style.display = "flex";

    const response = await fetch("/api/contribute", {
      method: "POST",
      body: formData,
    });

    document.getElementById("loading").style.display = "none";

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `Server error: ${response.status}`);
    }

    // Update contribution count
    userContributions++;
    localStorage.setItem("userContributions", userContributions);
    document.getElementById("contributions-count").innerText =
      userContributions;

    // Update total count
    const totalCount =
      parseInt(document.getElementById("total-count").innerText) + 1;
    document.getElementById("total-count").innerText = totalCount;

    // Show success message
    document.getElementById("success-message").style.display = "block";
    document.getElementById("status").innerText = "Contribution successful!";

    // Get a new sign prompt
    getNextSign();
  } catch (error) {
    console.error("Error processing video:", error);
    document.getElementById("status").innerText = `Error: ${
      error.message || "Could not process video"
    }`;
    document.getElementById("loading").style.display = "none";
  }
}

// Initialize when page loads
window.addEventListener("load", async () => {
  await initializeCamera();
  setupCameraControls();
  await fetchSignsData();
  await getNextSign();
});
