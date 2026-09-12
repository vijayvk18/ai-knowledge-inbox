/**
 * Add a note or a URL. Two modes, one form - the only difference is which
 * field is shown and which endpoint the submit hits.
 */
import { useState } from 'react';

export const AddItemForm = ({ onAddNote, onAddUrl }) => {
  const [mode, setMode] = useState('note');
  const [note, setNote] = useState('');
  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (event) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === 'note') {
        await onAddNote(note.trim(), title.trim() || undefined);
        setNote('');
      } else {
        await onAddUrl(url.trim(), title.trim() || undefined);
        setUrl('');
      }
      setTitle('');
    } catch (err) {
      setError(err);
    } finally {
      setSubmitting(false);
    }
  };

  const canSubmit = !submitting && (mode === 'note' ? note.trim().length > 0 : url.trim().length > 0);

  return (
    <form className="card" onSubmit={submit}>
      <div className="card-head">
        <h2>Save something</h2>
        <div className="tabs" role="tablist">
          {['note', 'url'].map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={mode === value}
              className={mode === value ? 'tab tab-active' : 'tab'}
              onClick={() => {
                setMode(value);
                setError(null);
              }}
            >
              {value === 'note' ? 'Note' : 'URL'}
            </button>
          ))}
        </div>
      </div>

      {mode === 'note' ? (
        <label className="field">
          <span className="field-label">Note text</span>
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Paste or type anything worth remembering..."
            rows={5}
          />
        </label>
      ) : (
        <label className="field">
          <span className="field-label">Page URL</span>
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder="https://example.com/article"
          />
          <span className="field-hint">The server fetches the page and indexes its readable text.</span>
        </label>
      )}

      <label className="field">
        <span className="field-label">
          Title <span className="muted">(optional)</span>
        </span>
        <input
          type="text"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder={mode === 'note' ? 'Defaults to the first line' : "Defaults to the page's title"}
        />
      </label>

      {error && (
        <p className="error" role="alert">
          {error.message}
          {error.requestId && <span className="muted"> (request {error.requestId.slice(0, 8)})</span>}
        </p>
      )}

      <button type="submit" className="primary" disabled={!canSubmit}>
        {submitting ? 'Saving...' : mode === 'note' ? 'Save note' : 'Fetch and save'}
      </button>
    </form>
  );
};
