import React, { useEffect, useState } from "react";
import PredictionForm from "../components/PredictionForm";
import PredictionResult from "../components/PredictionResult";

import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend
} from "chart.js";

import "../styles/predict.css";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend
);

export default function Predict({ setPrediction, prediction }) {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5001/logs")
      .then(res => res.json())
      .then(data => {
        const sorted = data.sort(
          (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
        );
        setLogs(sorted);
      })
      .catch(err => console.error(err));
  }, [prediction]);

  // =============================
  // GROUP DATA BY DAY
  // =============================

  const dailyMap = {};

  logs.forEach(item => {
    const date = new Date(item.timestamp).toISOString().split("T")[0]; // YYYY-MM-DD

    if (!dailyMap[date]) {
      dailyMap[date] = [];
    }

    dailyMap[date].push(item.prediction);
  });

  const dailyLabels = Object.keys(dailyMap).sort();

  const dailyAverages = dailyLabels.map(date => {
  let values = dailyMap[date];

  // If more than 2 values, remove extreme min & max
  if (values.length > 2) {
    values = [...values].sort((a, b) => a - b);
    values = values.slice(1, values.length - 1); // remove min & max
  }

  const avg =
    values.reduce((sum, val) => sum + val, 0) / values.length;

  return avg;
});


  // =============================
  // KPI CALCULATIONS (DAILY)
  // =============================

  const latest =
    dailyAverages.length > 0
      ? dailyAverages[dailyAverages.length - 1]
      : 0;

  const previous =
    dailyAverages.length > 1
      ? dailyAverages[dailyAverages.length - 2]
      : 0;

  const trend =
    latest > previous
      ? "⬆ Increasing"
      : latest < previous
      ? "⬇ Decreasing"
      : "➡ Stable";

  // Dynamic color based on trend
  const lineColor =
    latest > previous ? "#2e7d32" :
    latest < previous ? "#d32f2f" :
    "#007bff";

  // =============================
  // CHART DATA
  // =============================

  const chartData = {
    labels: dailyLabels,
    datasets: [
      {
        label: "Daily Avg AC Power Output (kW)",
        data: dailyAverages,
        borderColor: lineColor,
        backgroundColor: "rgba(0,123,255,0.15)",
        borderWidth: 3,
        pointRadius: 5,
        tension: 0.3
      }
    ]
  };

  return (
    <div className="dashboard-page">

      {/* KPI SECTION */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <h4>Latest Daily Output</h4>
          <p>{latest ? `${latest.toFixed(2)} kW` : "—"}</p>
        </div>

        <div className="kpi-card">
          <h4>Daily Trend</h4>
          <p>{trend}</p>
        </div>

        <div className="kpi-card">
          <h4>Total Predictions</h4>
          <p>{logs.length}</p>
        </div>
      </div>

      {/* MAIN GRID */}
      <div className="dashboard-grid">
        <div className="form-section card">
          <h3>Enter Solar Plant Data</h3>
          <PredictionForm setPrediction={setPrediction} />
          <PredictionResult prediction={prediction} />

        </div>

        <div className="chart-section card">
          <h3>Daily Power Output Trend</h3>

          {dailyLabels.length > 0 ? (
            <div style={{ height: "350px", width: "100%" }}>
              <Line
                data={chartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {
                    mode: "index",
                    intersect: false
                  },
                  plugins: {
                    tooltip: {
                      callbacks: {
                        label: function(context) {
                          return `Output: ${context.raw.toFixed(2)} kW`;
                        }
                      }
                    },
                    legend: {
                      labels: {
                        color: "#1a2b49",
                        font: { weight: "600" }
                      }
                    }
                  },
                  scales: {
                    x: {
                      ticks: {
                        color: "#444",
                        maxRotation: 45,
                        minRotation: 45
                      },
                      grid: {
                        color: "rgba(0,0,0,0.05)"
                      }
                    },
                    y: {
                      beginAtZero: true,
                      ticks: {
                        color: "#444",
                        callback: function(value) {
                          return value + " kW";
                        }
                      },
                      grid: {
                        color: "rgba(0,0,0,0.05)"
                      }
                    }
                  }
                }}
              />
            </div>
          ) : (
            <p>No data available yet.</p>
          )}
        </div>
      </div>

      {/* HISTORY TABLE */}
      <div className="history-section card">
        <h3>Prediction History</h3>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Irradiance</th>
              <th>Temp</th>
              <th>Output (kW)</th>
            </tr>
          </thead>
          <tbody>
            {logs.slice(-5).map((item) => (
              <tr key={item.id}>
                <td>
                  {new Date(item.timestamp).toISOString().split("T")[0]}
                </td>
                <td>{item.irradiance}</td>
                <td>{item.temp}</td>
                <td>{item.prediction.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}
