/**
 * App shell: two columns, one state owner per concern.
 *
 * `useItems` owns the inbox, `useAsk` owns the current question. Nothing is
 * shared between them except the item count, so no global store is warranted.
 */
import { useEffect, useState } from 'react';
import { AddItemForm } from './components/AddItemForm.jsx';
import { AskPanel } from './components/AskPanel.jsx';
import { ItemList } from './components/ItemList.jsx';
import { useAsk } from './hooks/useAsk.js';
import { useItems } from './hooks/useItems.js';
import { api } from './lib/api.js';

/** Which providers are actually live - the answer to most "why is it doing that" questions. */
const ProviderBar = ({ health }) => {
  if (!health) return null;
  const { embeddings, answers } = health.providers;

  return (
    <div className="providers">
      <span title="Embedding model used for indexing and search">
        embeddings: <strong>{embeddings.provider}</strong> ({embeddings.model})
      </span>
      <span title="Model used to write the answer">
        answers: <strong>{answers.provider}</strong> ({answers.model})
      </span>
      <span>
        index: <strong>{health.index.chunks}</strong> chunks / {health.index.items} items
      </span>
      {embeddings.provider === 'local' && (
        <span className="warn" title="Set OPENAI_API_KEY for semantic embeddings">
          lexical matching only
        </span>
      )}
    </div>
  );
};

export default function App() {
  const { items, counts, loading, error, addNote, addUrl, retry, remove } = useItems();
  const { result, asking, error: askError, ask } = useAsk();
  const [health, setHealth] = useState(null);

  // Refreshed alongside the item list so the index counter stays honest.
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, [items]);

  return (
    <div className="app">
      <header className="app-head">
        <div>
          <h1>AI Knowledge Inbox</h1>
          <p className="muted">Save notes and pages, then ask questions answered only from what you saved.</p>
        </div>
        <ProviderBar health={health} />
      </header>

      <main className="columns">
        <div className="column">
          <AddItemForm onAddNote={addNote} onAddUrl={addUrl} />
          <section className="card">
            <div className="card-head">
              <h2>Saved items</h2>
            </div>
            <ItemList
              items={items}
              counts={counts}
              loading={loading}
              error={error}
              onRetry={retry}
              onRemove={remove}
            />
          </section>
        </div>

        <div className="column">
          <AskPanel
            result={result}
            asking={asking}
            error={askError}
            onAsk={ask}
            indexedItems={counts.ready ?? 0}
          />
        </div>
      </main>
    </div>
  );
}
