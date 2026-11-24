
import { useState } from "react";
import "./Recommendations.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const createJob = () => ({
  app: "",
  func: "",
  start_timestamp: "",
  end_timestamp: "",
  duration: "",
  workload_type: "immediate",
});

const createCarbonReading = () => ({
  minute: "",
  carbon_intensity_lifecycle: "",
});

function Recommendations() {
  const [jobs, setJobs] = useState([createJob()]);
  const [carbonData, setCarbonData] = useState([
    createCarbonReading(),
  ]);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateJob = (index, field, value) => {
    setJobs((previous) =>
      previous.map((job, i) =>
        i === index ? { ...job, [field]: value } : job
      )
    );
  };

  const updateCarbon = (index, field, value) => {
    setCarbonData((previous) =>
      previous.map((reading, i) =>
        i === index ? { ...reading, [field]: value } : reading
      )
    );
  };

  const addJob = () => setJobs((previous) => [...previous, createJob()]);

  const removeJob = (index) => {
    setJobs((previous) => previous.filter((_, i) => i !== index));
  };

  const addCarbonReading = () => {
    setCarbonData((previous) => [
      ...previous,
      createCarbonReading(),
    ]);
  };

  const removeCarbonReading = (index) => {
    setCarbonData((previous) =>
      previous.filter((_, i) => i !== index)
    );
  };

  const validateData = () => {
    if (jobs.length === 0 || carbonData.length === 0) {
      return "Add at least one workload job and one carbon reading.";
    }

    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i];

      if (
        !job.app.trim() ||
        !job.func.trim() ||
        job.start_timestamp === "" ||
        job.end_timestamp === "" ||
        job.duration === ""
      ) {
        return `Please complete all fields in workload job ${i + 1}.`;
      }

      const start = Number(job.start_timestamp);
      const end = Number(job.end_timestamp);
      const duration = Number(job.duration);

      if (
        !Number.isFinite(start) ||
        !Number.isFinite(end) ||
        !Number.isFinite(duration) ||
        duration < 0 ||
        end < start
      ) {
        return `Invalid timestamps or duration in workload job ${i + 1}.`;
      }
    }

    for (let i = 0; i < carbonData.length; i++) {
      const reading = carbonData[i];

      if (
        reading.minute === "" ||
        reading.carbon_intensity_lifecycle === ""
      ) {
        return `Please complete all fields in carbon reading ${i + 1}.`;
      }

      if (
        !Number.isFinite(Number(reading.minute)) ||
        !Number.isFinite(Number(reading.carbon_intensity_lifecycle))
      ) {
        return `Invalid numeric value in carbon reading ${i + 1}.`;
      }
    }

    return "";
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);

    const validationError = validateData();

    if (validationError) {
      setError(validationError);
      return;
    }

    const payload = {
      workload: jobs.map((job) => ({
        app: job.app.trim(),
        func: job.func.trim(),
        start_timestamp: Number(job.start_timestamp),
        end_timestamp: Number(job.end_timestamp),
        duration: Number(job.duration),
        workload_type: job.workload_type,
      })),

      carbon_data: carbonData.map((reading) => ({
        minute: Number(reading.minute),
        carbon_intensity_lifecycle: Number(
          reading.carbon_intensity_lifecycle
        ),
      })),
    };

    try {
      setLoading(true);

      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || "Prediction request failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Cannot connect to the Python API. Check that FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setJobs([createJob()]);
    setCarbonData([createCarbonReading()]);
    setResult(null);
    setError("");
  };

  return (
    <div className="recommendations-page">
      <div className="recommendations-header">
        <div>
          <h2>ML Recommendations</h2>
          <p>
            Enter workload and carbon data to generate an intelligent
            scheduling recommendation.
          </p>
        </div>

        <span className="model-status">
          Random Forest Models
        </span>
      </div>

      <form onSubmit={handleSubmit}>
        {/* WORKLOAD INPUT */}

        <section className="input-section">
          <div className="section-heading">
            <div>
              <h3>Workload Information</h3>
              <p>
                Enter the jobs that need to be scheduled.
                Timestamps and durations are in seconds.
              </p>
            </div>

            <button
              type="button"
              className="add-button"
              onClick={addJob}
            >
              + Add Job
            </button>
          </div>

          <div className="table-wrapper">
            <table className="recommendation-table">
              <thead>
                <tr>
                  <th>App</th>
                  <th>Function</th>
                  <th>Start (sec)</th>
                  <th>End (sec)</th>
                  <th>Duration (sec)</th>
                  <th>Workload Type</th>
                  <th>Action</th>
                </tr>
              </thead>

              <tbody>
                {jobs.map((job, index) => (
                  <tr key={index}>
                    <td>
                      <input
                        value={job.app}
                        onChange={(e) =>
                          updateJob(index, "app", e.target.value)
                        }
                        placeholder="app_1"
                      />
                    </td>

                    <td>
                      <input
                        value={job.func}
                        onChange={(e) =>
                          updateJob(index, "func", e.target.value)
                        }
                        placeholder="function_1"
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        min="0"
                        value={job.start_timestamp}
                        onChange={(e) =>
                          updateJob(
                            index,
                            "start_timestamp",
                            e.target.value
                          )
                        }
                        placeholder="60"
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        min="0"
                        value={job.end_timestamp}
                        onChange={(e) =>
                          updateJob(
                            index,
                            "end_timestamp",
                            e.target.value
                          )
                        }
                        placeholder="65"
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        min="0"
                        value={job.duration}
                        onChange={(e) =>
                          updateJob(index, "duration", e.target.value)
                        }
                        placeholder="5"
                      />
                    </td>

                    <td>
                      <select
                        value={job.workload_type}
                        onChange={(e) =>
                          updateJob(
                            index,
                            "workload_type",
                            e.target.value
                          )
                        }
                      >
                        <option value="immediate">Immediate</option>
                        <option value="delay_tolerant">
                          Delay Tolerant
                        </option>
                        <option value="other">Other</option>
                      </select>
                    </td>

                    <td>
                      <button
                        type="button"
                        className="remove-button"
                        onClick={() => removeJob(index)}
                        disabled={jobs.length === 1}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="table-footer">
            Total workload jobs: <strong>{jobs.length}</strong>
          </div>
        </section>

        {/* CARBON INPUT */}

        <section className="input-section">
          <div className="section-heading">
            <div>
              <h3>Carbon Intensity Data</h3>
              <p>
                Enter carbon intensity readings. Use the same
                lifecycle-intensity units as your training dataset.
              </p>
            </div>

            <button
              type="button"
              className="add-button"
              onClick={addCarbonReading}
            >
              + Add Reading
            </button>
          </div>

          <div className="table-wrapper">
            <table className="recommendation-table carbon-table">
              <thead>
                <tr>
                  <th>Minute</th>
                  <th>Carbon Intensity (Lifecycle)</th>
                  <th>Action</th>
                </tr>
              </thead>

              <tbody>
                {carbonData.map((reading, index) => (
                  <tr key={index}>
                    <td>
                      <input
                        type="number"
                        min="0"
                        value={reading.minute}
                        onChange={(e) =>
                          updateCarbon(index, "minute", e.target.value)
                        }
                        placeholder="0"
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        step="any"
                        value={reading.carbon_intensity_lifecycle}
                        onChange={(e) =>
                          updateCarbon(
                            index,
                            "carbon_intensity_lifecycle",
                            e.target.value
                          )
                        }
                        placeholder="300"
                      />
                    </td>

                    <td>
                      <button
                        type="button"
                        className="remove-button"
                        onClick={() => removeCarbonReading(index)}
                        disabled={carbonData.length === 1}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="table-footer">
            Total carbon readings: <strong>{carbonData.length}</strong>
          </div>
        </section>

        {/* ACTIONS */}

        <div className="form-actions">
          <button
            type="button"
            className="reset-button"
            onClick={resetForm}
          >
            Reset Form
          </button>

          <button
            type="submit"
            className="predict-button"
            disabled={loading}
          >
            {loading ? "Generating Recommendation..." : "Generate Recommendation"}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* PREDICTION RESULT */}

      {result && (
        <section className="prediction-result">
          <div className="result-heading">
            <div>
              <span className="result-eyebrow">Prediction Complete</span>
              <h3>Recommended Scheduling Strategy</h3>
            </div>

            <span className="success-label">Success</span>
          </div>

          <div className="result-cards">
            <div className="result-card strategy-card">
              <span>Recommended Strategy</span>
              <h2>{result.strategy?.replaceAll("_", " ")}</h2>
            </div>

            <div className="result-card">
              <span>Recommended Deadline</span>
              <h2>{result.deadline_minutes} min</h2>
            </div>

            <div className="result-card">
              <span>ML Model</span>
              <h2>{result.model_type || "Random Forest"}</h2>
            </div>

            <div className="result-card">
              <span>Features Used</span>
              <h2>{result.feature_count}/32</h2>
            </div>
          </div>

          <div className="result-details">
            <p>
              <strong>Models retrained:</strong>{" "}
              {result.models_retrained ? "Yes" : "No"}
            </p>

            <p>
              <strong>Workload records:</strong> {jobs.length}
            </p>

            <p>
              <strong>Carbon readings:</strong> {carbonData.length}
            </p>

            {result.message && <p>{result.message}</p>}
          </div>

          <div className="derived-features">
            <h4>Extracted ML Features</h4>

            <div className="feature-grid">
              {Object.entries(result.derived_features || {}).map(
                ([key, value]) => (
                  <div className="feature-item" key={key}>
                    <span>{key.replaceAll("_", " ")}</span>
                    <strong>
                      {typeof value === "number"
                        ? Number(value.toFixed(4))
                        : value}
                    </strong>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="comparison-notice">
            <h4>Scheduling Strategy Comparison</h4>
            {result.comparisons ? (
              <p>
                Comparison data received from the API. The comparison
                results can now be connected to the results visualization.
              </p>
            ) : (
              <p>
                Strategy-level energy, carbon and cost comparisons are
                not yet available. The current API predicts the strategy
                and deadline only; it does not execute the scheduling
                simulation.
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

export default Recommendations;
