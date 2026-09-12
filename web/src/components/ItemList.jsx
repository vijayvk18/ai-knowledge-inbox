/**
 * The saved-items list.
 *
 * Status is the point of this view: ingestion is async, so every row says where
 * it is in the pipeline, and a failed row shows why plus a way to retry.
 */
const STATUS_LABEL = {
  pending: 'Queued',
  processing: 'Indexing',
  ready: 'Indexed',
  failed: 'Failed',
};

const relativeTime = (iso) => {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

const ItemRow = ({ item, onRetry, onRemove }) => (
  <li className={`item item-${item.status}`}>
    <div className="item-head">
      <span className={`badge badge-${item.status}`}>{STATUS_LABEL[item.status]}</span>
      <span className="item-type">{item.sourceType}</span>
      <h3 className="item-title">{item.title || 'Untitled'}</h3>
    </div>

    {item.url && (
      <a className="item-url" href={item.url} target="_blank" rel="noreferrer noopener">
        {item.url}
      </a>
    )}

    {item.preview && <p className="item-preview">{item.preview}</p>}

    {item.error && (
      <p className="error" role="alert">
        {item.error.message} <span className="muted">({item.error.code})</span>
      </p>
    )}

    <div className="item-foot">
      <span className="muted">
        {relativeTime(item.createdAt)}
        {item.status === 'ready' && ` - ${item.chunkCount} chunk${item.chunkCount === 1 ? '' : 's'}, ${item.charCount.toLocaleString()} chars`}
      </span>
      <span className="item-actions">
        {item.status === 'failed' && (
          <button type="button" className="link" onClick={() => onRetry(item.id)}>
            Retry
          </button>
        )}
        <button type="button" className="link danger" onClick={() => onRemove(item.id)}>
          Delete
        </button>
      </span>
    </div>
  </li>
);

export const ItemList = ({ items, counts, loading, error, onRetry, onRemove }) => {
  if (loading) return <p className="muted">Loading saved items...</p>;
  if (error) {
    return (
      <p className="error" role="alert">
        {error.message}
      </p>
    );
  }
  if (items.length === 0) {
    return <p className="muted">Nothing saved yet. Add a note or a URL to build your inbox.</p>;
  }

  const summary = Object.entries(counts)
    .map(([status, count]) => `${count} ${STATUS_LABEL[status].toLowerCase()}`)
    .join(' - ');

  return (
    <>
      <p className="muted list-summary">{summary}</p>
      <ul className="item-list">
        {items.map((item) => (
          <ItemRow key={item.id} item={item} onRetry={onRetry} onRemove={onRemove} />
        ))}
      </ul>
    </>
  );
};
