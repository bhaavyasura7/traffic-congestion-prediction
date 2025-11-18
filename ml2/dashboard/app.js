// dashboard/app.js
// Loads prediction and model performance data, and updates the dashboard UI

document.addEventListener("DOMContentLoaded", function () {
  // Video upload integration
  const videoInput = document.getElementById("videoInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const videoStatus = document.getElementById("videoStatus");
  // const loadingScreen = document.getElementById("loadingScreen");

  uploadBtn.addEventListener("click", function () {
    const file = videoInput.files[0];
    console.log("[UPLOAD] Selected file:", file);

    if (!file) {
      alert("Please select a video file to upload.");
      return;
    }

    // ✅ SHOW loader before uploading
    // loadingScreen.style.display = "flex";

    videoStatus.textContent = "Uploading...";
    videoStatus.style.color = "#ffd600";
    uploadBtn.disabled = true;

    const formData = new FormData();
    formData.append("video", file);
    console.log("[UPLOAD] FormData keys:", Array.from(formData.keys()));

    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        console.log("[UPLOAD] Response status:", response.status);
        return response.json();
      })
      .then((data) => {
        console.log("[UPLOAD] Response data:", data);
        // Hide loading screen and enable button
        // loadingScreen.style.display = "none";
        uploadBtn.disabled = false;
        // Show status
        videoStatus.textContent = data.message || "Processing complete!";
        videoStatus.style.color = "#00ff00";
        // Auto-update dashboard with new results
        if (data.model_performance) {
          try {
            const modelPerfJson = JSON.parse(data.model_performance);
            renderPerformanceMetrics(modelPerfJson);
          } catch (e) {
            console.log("Error parsing model_performance:", e);
          }
        }
        if (data.prediction) {
          try {
            const predictionJson = JSON.parse(data.prediction);
            renderPredictionCards(predictionJson);
            renderCongestionChart(predictionJson);
            updateTopStats(predictionJson);
          } catch (e) {
            console.log("Error parsing prediction:", e);
          }
        }
      })
      .catch((error) => {
        console.error("[UPLOAD] Network or fetch error:", error);
        // loadingScreen.style.display = "none";
        videoStatus.textContent = "Error uploading video.";
        videoStatus.style.color = "#ff0000";
        uploadBtn.disabled = false;
      });
  });

  // Navigation toggle
  const predictionBtn = document.getElementById("predictionBtn");
  const predictionSection = document.querySelector(".prediction-section");
  const chartsRow = document.querySelector(".charts-row");
  // const monitorSection = document.querySelector(".monitor-section");

  predictionBtn.addEventListener("click", function () {
    predictionBtn.classList.add("active");
    predictionSection.style.display = "block";
    chartsRow.style.display = "flex";
    // if (monitorSection) monitorSection.style.display = "none";
  });

  // Load prediction cards
  fetch("data/prediction.json")
    .then((res) => res.json())
    .then((data) => {
      renderPredictionCards(data);
      renderCongestionChart(data);
      updateTopStats(data);
    });
  // Update top stats (congestion % and vehicles)
  function updateTopStats(data) {
    if (!Array.isArray(data) || data.length === 0) return;
    // Find current hour prediction (predict_hour === 0)
    const current = data.find((p) => p.predict_hour === 0) || data[0];
    // Update current hour confidence and congestion level beside upload
    document.getElementById("currentConfidence").textContent =
      `Confidence: ${current.confidence !== undefined ? (current.confidence * 100).toFixed(2) : "--"}`;
    document.getElementById("currentCongestion").textContent =
      `Congestion Level: ${current.congestion_level || "--"}`;
  }

  // Load model performance
  fetch("data/model_performance.json")
    .then((res) => res.json())
    .then((data) => {
      renderPerformanceMetrics(data);
    });

  // Prediction Cards
  function renderPredictionCards(data) {
    const container = document.getElementById("predictionCards");
    container.innerHTML = "";
    data.forEach((pred) => {
      let color = "green";
      if (
        pred.congestion_level === "Very High" ||
        pred.congestion_level === "High"
      )
        color = "red";
      else if (pred.congestion_level === "Moderate") color = "yellow";
      let note = getCardNote(pred.congestion_level, pred.predict_hour);
      container.innerHTML += `
                <div class="card ${color}">
                    <div class="card-title">
                      ${pred.predict_hour === 0
          ? "Current"
          : "next " + pred.predict_hour + " Hr"
        }
                    </div>
                    <div class="card-detail">Traffic level : <b>${pred.congestion_level}</b></div>
                    <div class="card-detail">Confidence : <b>${pred.confidence !== undefined ? (pred.confidence * 100).toFixed(2) : "--"}</b></div>
                    <div class="card-note">${note}</div>
                </div>
            `;
    });
  }
  function getWaitTime(level) {
    switch (level) {
      case "Very High":
        return "8-12 min";
      case "High":
        return "8-10 min";
      case "Moderate":
        return "7-9 min";
      case "Light":
        return "5-6 min";
      case "Very Light":
        return "4-5 min";
      default:
        return "N/A";
    }
  }
  function getCardNote(level, hour) {
    if (level === "Very High")
      return hour === 0
        ? "Heaviest traffic office day - Avoid if possible"
        : "Traffic expected - Consider alternate routes";
    if (level === "High")
      return "Peak traffic expected - Consider alternate routes";
    if (level === "Moderate") return "Traffic Clearing up nicely";
    if (level === "Light") return "Good time to travel";
    if (level === "Very Light") return "Smooth traffic flow";
    return "";
  }

  // Congestion Chart
  function renderCongestionChart(data) {
    const ctx = document.getElementById("congestionChart").getContext("2d");
    const labels = data.map((p) => `+${p.predict_hour}h`);
    const values = data.map((p) => p.confidence !== undefined ? (p.confidence * 100).toFixed(2) : null);
    new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Confidence %",
            data: values,
            borderColor: "#1976d2",
            backgroundColor: "rgba(25,118,210,0.2)",
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: "#ffd600",
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "Confidence (%)" },
          },
          x: { title: { display: true, text: "Hour" } },
        },
      },
    });
  }

  // Model Performance Metrics
  function renderPerformanceMetrics(data) {
    // Use actual values from model_performance.json
    const perf = data.model_performance || {};
    const metrics = [
      { label: "F1 Score", value: perf.f1_score, color: "green" },
      { label: "Precision Score", value: perf.precision_score, color: "blue" },
      { label: "Recall Score", value: perf.recall_score, color: "purple" },
    ];
    const container = document.getElementById("performanceMetrics");
    container.innerHTML = "";
    metrics.forEach((m) => {
      container.innerHTML += `
                <div class="metric-row">
                    <div class="metric-label">${m.label}</div>
                    <div class="metric-bar">
                        <div class="metric-bar-inner ${m.color
        }" style="width:${(m.value * 100).toFixed(1)}%"></div>
                    </div>
                    <span class="metric-value">${(m.value * 100).toFixed(
          2
        )}%</span>
                </div>
            `;
    });
  }

  // Traffic flow chart (dummy data)
  if (window.Chart) {
    const flowCtx = document.getElementById("flowChart").getContext("2d");
    new Chart(flowCtx, {
      type: "line",
      data: {
        labels: ["07:00", "09:00", "11:00", "13:00", "15:00"],
        datasets: [
          {
            label: "Vehicles",
            data: [100, 80, 60, 90, 95],
            borderColor: "#43a047",
            backgroundColor: "rgba(67,160,71,0.2)",
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: "#ffd600",
          },
          {
            label: "Congestion",
            data: [25, 30, 28, 27, 26],
            borderColor: "#1976d2",
            backgroundColor: "rgba(25,118,210,0.2)",
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: "#ffd600",
          },
        ],
      },
      options: {
        plugins: { legend: { display: true } },
        scales: {
          y: { min: 0, max: 100 },
          x: {},
        },
      },
    });
  }
});
