
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://10.1.34.90:8000";

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data.error || {});

    throw new Error(detail || `Backend request failed (${response.status})`);
  }

  return data;
}

export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return readResponse(response);
}

export async function matchEvaluatorTrip(tripId) {
  const response = await fetch(`${API_BASE_URL}/match-evaluator-trip`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      trip_id: tripId,
      search_radius_m: 100,
      max_candidates: 8,
      transition_scale_m: 50,
    }),
  });

  return readResponse(response);
}
