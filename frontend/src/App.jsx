import {
  checkBackendHealth,
  matchEvaluatorTrip,
} from "./services/api";
import { useEffect, useMemo, useState } from "react";
import {
  Activity, Upload, MapPin, Route, Database, FileText,
  RefreshCw, CheckCircle2, AlertCircle, Navigation,
  Layers, Clock3, Crosshair, Menu, X
} from "lucide-react";
import {
  MapContainer, TileLayer, CircleMarker, Polyline,
  Popup, useMap
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";



function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);

  if (lines.length < 2) {
    throw new Error("The CSV file has no GPS observations.");
  }

  const headers = lines[0].split(",").map(h => h.trim().toLowerCase());

  const lonIndex = headers.findIndex(h =>
    ["lon", "lng", "longitude"].includes(h)
  );

  const latIndex = headers.findIndex(h =>
    ["lat", "latitude"].includes(h)
  );

  if (lonIndex === -1 || latIndex === -1) {
    throw new Error(
      "CSV must contain longitude/lon and latitude/lat columns."
    );
  }

  const points = lines.slice(1).map((line, index) => {
    const values = line.split(",").map(v => v.trim());

    const lat = Number(values[latIndex]);
    const lng = Number(values[lonIndex]);

    if (
      !Number.isFinite(lat) ||
      !Number.isFinite(lng) ||
      lat < -90 || lat > 90 ||
      lng < -180 || lng > 180
    ) {
      return null;
    }

    return { lat, lng, index: index + 1 };
  }).filter(Boolean);

  if (!points.length) {
    throw new Error("No valid GPS coordinates were found.");
  }

  return points;
}


function FitBounds({ points }) {
  const map = useMap();

  useEffect(() => {
    if (points.length) {
      map.fitBounds(
        points.map(p => [p.lat, p.lng]),
        { padding: [45, 45], maxZoom: 16 }
      );
    }
  }, [map, points]);

  return null;
}


function StatCard({ icon: Icon, label, value, note }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <Icon size={18} />
      </div>

      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </div>
  );
}


export default function App() {
  const [file, setFile] = useState(null);
  const [points, setPoints] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [matchResult, setMatchResult] = useState(null);

  // Backend health state
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [modelName, setModelName] = useState("");
  const [backendError, setBackendError] = useState("");

  // Check Varchas's backend when the dashboard loads.
  async function refreshBackendHealth() {
    setBackendStatus("Checking...");
    setBackendError("");

    try {
      const data = await checkBackendHealth();

      if (data.model_loaded) {
        setBackendStatus("Connected");
        setModelName(data.model_name || "ML model loaded");
      } else {
        setBackendStatus("Model unavailable");
        setModelName("");
        setBackendError(
          data.model_error || "Backend is running, but its model is unavailable."
        );
      }
    } catch (err) {
      setBackendStatus("Disconnected");
      setModelName("");
      setBackendError(
        err.message || "Unable to connect to the backend."
      );
    }
  }

  useEffect(() => {
    refreshBackendHealth();
  }, []);

  const center = useMemo(
    () => points.length
      ? [points[0].lat, points[0].lng]
      : [20.5937, 78.9629],
    [points]
  );

  async function handleFile(event) {
    const selected = event.target.files?.[0];

    if (!selected) return;

    setError("");
    setFile(null);
    setPoints([]);
    setStatus("loading");

    try {
      if (!selected.name.toLowerCase().endsWith(".csv")) {
        throw new Error("Please upload a CSV file.");
      }

      const text = await selected.text();
      const parsed = parseCSV(text);

      setFile(selected);
      setPoints(parsed);
      setStatus("loaded");
    } catch (err) {
      setError(err.message || "Unable to read this file.");
      setStatus("error");
    }
  }

  function reset() {
    setFile(null);
    setPoints([]);
    setError("");
    setStatus("idle");
    setBusy(false);

    const input = document.getElementById("gps-file");

    if (input) input.value = "";
  }

  
async function runPrediction() {
  setError("");
  setMatchResult(null);

  if (backendStatus !== "Connected") {
    setError("Backend is not connected. Check Laptop 1.");
    return;
  }

  if (!selectedTrip) {
    setError("Please select an evaluator trip.");
    return;
  }

  setBusy(true);

  try {
    const data = await matchEvaluatorTrip(selectedTrip);

    setMatchResult(data);

    const mappedPoints = (data.observations || [])
      .filter(
        (item) =>
          Number.isFinite(Number(item.latitude)) &&
          Number.isFinite(Number(item.longitude))
      )
      .map((item, index) => ({
        lat: Number(item.latitude),
        lng: Number(item.longitude),
        index: index + 1,
        candidateRoadId: item.candidate_road_id,
        probability: item.match_probability,
      }));

    setPoints(mappedPoints);
    setStatus("loaded");

    if (!mappedPoints.length) {
      setError("The API returned no valid mapped observations.");
    }
  } catch (err) {
    setError(err.message || "Trajectory matching failed.");
  } finally {
    setBusy(false);
  }
}


  const backendConnected = backendStatus === "Connected";

  return (
    <div className="app-shell">

      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>

        <div className="brand">
          <div className="brand-icon">
            <Route size={23} />
          </div>

          <div>
            <strong>PathSnap</strong>
            <small>GPS INTELLIGENCE</small>
          </div>

          <button
            className="mobile-close"
            onClick={() => setMobileMenu(false)}
          >
            <X />
          </button>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <div className="nav-item active">
          <Layers size={18} /> Overview
        </div>

        <div className="nav-item">
          <MapPin size={18} /> Map matching
        </div>

        <div className="nav-item">
          <Database size={18} /> Dataset
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            Frontend online
          </div>

          <div className="sidebar-caption">
            PATHSNAP · HACKATHON PROJECT
          </div>
        </div>

      </aside>


      <main className="main-area">

        <header className="topbar">

          <button
            className="menu-button"
            onClick={() => setMobileMenu(!mobileMenu)}
          >
            <Menu />
          </button>

          <div className="breadcrumb">
            Workspace <span>/</span> Overview
          </div>

          <div className="topbar-right">
            <span className="live-dot" />
            Development mode
          </div>

        </header>


        <div className="page-content">

          <section className="welcome-row">

            <div>
              <div className="eyebrow">
                GPS TRAJECTORY ANALYTICS
              </div>

              <h1>
                Map Matching <span>Workspace</span>
              </h1>

              <p>
                Transform noisy GPS observations into meaningful road trajectories.
              </p>
            </div>

            <div className="project-chip">
              <Activity size={16} /> PathSnap ML
            </div>

          </section>


          <section className="stats-grid">

            <StatCard
              icon={MapPin}
              label="GPS OBSERVATIONS"
              value={points.length.toLocaleString()}
              note="Valid uploaded coordinates"
            />

            <StatCard
              icon={Route}
              label="TRAJECTORY STATUS"
              value={status === "loaded" ? "Loaded" : "Waiting"}
              note="GPS input processing"
            />

            <StatCard
              icon={FileText}
              label="INPUT FILE"
              value={file ? "CSV ready" : "None"}
              note={file ? file.name : "No file selected"}
            />

            <StatCard
              icon={Activity}
              label="ML BACKEND"
              value={backendStatus}
              note={modelName || backendError || "Checking backend connection..."}
            />

          </section>


          <section className="panel trip-selector-panel">

            <div className="panel-heading">

              <div>
                <div className="panel-kicker">
                  EVALUATION WORKSPACE
                </div>

                <h2>Select a trajectory</h2>
              </div>

              <Route size={20} />

            </div>

            <p className="trip-description">
              Select an evaluation trip provided by the backend to begin map matching.
            </p>

            <div className="trip-controls">

              
<select
  value={selectedTrip}
  onChange={(event) => setSelectedTrip(event.target.value)}
  className="trip-select"
>
  <option value="">Select an evaluator trip</option>
  {[
    "00000015",
    "00000017",
    "00000023",
    "00000030",
    "00000049",
    "00000059",
    "00000067",
    "00000073",
    "00000087",
    "00000094",
  ].map((trip) => (
    <option key={trip} value={trip}>
      Trip {trip}
    </option>
  ))}
</select>


              <button
                className="primary-button"
                disabled={!selectedTrip || busy}
                onClick={runPrediction}
              >
                <Navigation size={17} />
                Run map matching
              </button>

            </div>

            <p className="helper-text">
              The evaluator trip list will be populated when the trajectory API is integrated.
            </p>

          </section>


          <section className="workspace-grid">

            <div className="panel upload-panel">

              <div className="panel-heading">

                <div>
                  <div className="panel-kicker">STEP 01</div>
                  <h2>Upload trajectory</h2>
                </div>

                <Upload size={20} />

              </div>


              <label className="upload-zone" htmlFor="gps-file">

                <div className="upload-icon">
                  <Upload size={25} />
                </div>

                <strong>Choose your GPS CSV file</strong>

                <span>Click to browse your computer</span>

                <small>
                  CSV format · Requires latitude and longitude columns
                </small>

                <input
                  id="gps-file"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleFile}
                />

              </label>


              {file && (
                <div className="file-row">

                  <FileText size={19} />

                  <div className="file-info">
                    <strong>{file.name}</strong>

                    <small>
                      {(file.size / 1024).toFixed(1)} KB · {points.length} valid observations
                    </small>
                  </div>

                  <CheckCircle2 size={19} className="success-icon" />

                </div>
              )}


              {error && (
                <div className="error-message">
                  <AlertCircle size={17} />
                  {error}
                </div>
              )}


              <div className="button-row">

                <button
                  className="primary-button"
                  disabled={!points.length || busy}
                  onClick={runPrediction}
                >
                  <Navigation size={17} />
                  Run map matching
                </button>

                <button
                  className="secondary-button"
                  onClick={reset}
                >
                  <RefreshCw size={16} />
                  Reset
                </button>

              </div>

              <p className="helper-text">
                CSV visualisation is available. Prediction requires the full trajectory pipeline.
              </p>

            </div>


            <div className="panel map-panel">

              <div className="panel-heading">

                <div>
                  <div className="panel-kicker">STEP 02</div>
                  <h2>Trajectory map</h2>
                </div>

                <div className="map-badge">
                  <span className="status-dot" />
                  Interactive
                </div>

              </div>


              <div className="map-wrap">

                <MapContainer
                  center={center}
                  zoom={5}
                  scrollWheelZoom
                  className="leaflet-map"
                >

                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />

                  {points.length > 0 && (
                    <FitBounds points={points} />
                  )}

                  {points.length > 1 && (
                    <Polyline
                      positions={points.map(p => [p.lat, p.lng])}
                      pathOptions={{
                        color: "#26b7a4",
                        weight: 4,
                        opacity: 0.85
                      }}
                    />
                  )}

                  {points.map((p, i) => (
                    <CircleMarker
                      key={p.index}
                      center={[p.lat, p.lng]}
                      radius={
                        i === 0 || i === points.length - 1 ? 7 : 4
                      }
                      pathOptions={{
                        color: "#ffffff",
                        weight: 1,
                        fillColor:
                          i === 0
                            ? "#27c99a"
                            : i === points.length - 1
                              ? "#f5a64a"
                              : "#3287f6",
                        fillOpacity: 1
                      }}
                    >
                      <Popup>
                        GPS observation #{p.index}
                        <br />
                        Lat: {p.lat}
                        <br />
                        Lon: {p.lng}
                      </Popup>
                    </CircleMarker>
                  ))}

                </MapContainer>


                {points.length === 0 && (
                  <div className="map-empty">

                    <Crosshair size={29} />

                    <strong>
                      Your trajectory will appear here
                    </strong>

                    <span>
                      Upload a GPS CSV file to visualize observations.
                    </span>

                  </div>
                )}

              </div>


              <div className="map-legend">

                <span>
                  <i className="legend-start" /> Start
                </span>

                <span>
                  <i className="legend-point" /> GPS observations
                </span>

                <span>
                  <i className="legend-end" /> End
                </span>

              </div>

            </div>

          </section>


          <section className="panel results-panel">

            <div className="panel-heading">

              <div>
                <div className="panel-kicker">STEP 03</div>
                <h2>Analysis & prediction results</h2>
              </div>

              <span className="pending-badge">
                {backendConnected ? "Backend connected" : "Awaiting backend"}
              </span>

            </div>


            
{matchResult ? (
  <div className="results-content">
    <div className="results-empty">
      <div className="results-icon">
        <CheckCircle2 size={23} />
      </div>
      <div>
        <strong>Trajectory matching completed</strong>
        <p>
          Model: {matchResult.model_name || modelName}
        </p>
        <p>
          Matched roads: {matchResult.matched_roads?.length ?? 0}
        </p>
        <p>
          Unmatched observations:{" "}
          {matchResult.unmatched_observation_indices?.length ?? 0}
        </p>
      </div>
    </div>

    <h3>Observation-level predictions</h3>

    <div style={{ overflowX: "auto" }}>
      <table className="results-table">
        <thead>
          <tr>
            <th>Observation</th>
            <th>Road ID</th>
            <th>Match probability</th>
          </tr>
        </thead>
        <tbody>
          {(matchResult.observations || []).slice(0, 50).map((item) => (
            <tr key={item.observation_id}>
              <td>{item.observation_id}</td>
              <td>{item.candidate_road_id ?? "N/A"}</td>
              <td>
                {Number.isFinite(Number(item.match_probability))
                  ? `${(Number(item.match_probability) * 100).toFixed(2)}%`
                  : "N/A"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    <p className="helper-text">
      Showing up to 50 observations. Matched road IDs are returned by the backend.
    </p>
  </div>
) : (
  <div className="results-empty">
    <div className="results-icon">
      <Activity size={23} />
    </div>
    <div>
      <strong>
        {backendConnected
          ? "ML model connected successfully"
          : "Waiting for ML backend"}
      </strong>
      <p>
        {backendConnected
          ? "Select an evaluator trip and run map matching to see predictions."
          : backendError || "Checking the FastAPI backend connection."}
      </p>
      <button
        className="secondary-button"
        onClick={refreshBackendHealth}
      >
        <RefreshCw size={15} />
        Check connection again
      </button>
    </div>
  </div>
)}

          </section>


          <footer className="footer">

            <span>
              PathSnap · GPS Trajectory Map Matching
            </span>

            <span>
              <Clock3 size={13} /> Development build
            </span>

          </footer>

        </div>

      </main>

    </div>
  );
}
