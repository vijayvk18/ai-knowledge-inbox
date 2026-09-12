/**
 * The only module that talks to the backend.
 *
 * Every failure becomes an ApiError carrying the server's own `code`, `message`
 * and `requestId`, so components render the message the API wrote instead of
 * inventing their own, and a user can quote the request id in a bug report.
 */
export class ApiError extends Error {
  constructor({ message, code, status, requestId, details }) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.details = details;
  }
}

const request = async (path, { method = 'GET', body, signal } = {}) => {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError({
      message: 'Cannot reach the server. Is the backend running on port 8787?',
      code: 'network_error',
      status: 0,
    });
  }

  if (response.status === 204) return null;

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError({
      message: payload?.error?.message ?? `Request failed with status ${response.status}`,
      code: payload?.error?.code ?? 'unknown_error',
      status: response.status,
      requestId: payload?.error?.requestId,
      details: payload?.error?.details,
    });
  }

  return payload;
};

export const api = {
  health: () => request('/health'),
  listItems: ({ limit = 50 } = {}) => request(`/items?limit=${limit}`),
  ingestNote: (content, title) => request('/ingest', { method: 'POST', body: { type: 'note', content, ...(title ? { title } : {}) } }),
  ingestUrl: (url, title) => request('/ingest', { method: 'POST', body: { type: 'url', url, ...(title ? { title } : {}) } }),
  retryItem: (id) => request(`/items/${id}/retry`, { method: 'POST' }),
  deleteItem: (id) => request(`/items/${id}`, { method: 'DELETE' }),
  ask: (question, topK, signal) => request('/query', { method: 'POST', body: { question, topK }, signal }),
};
