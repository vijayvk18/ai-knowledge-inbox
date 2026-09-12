/**
 * The question box.
 *
 * `answered: false` is not an error - an empty inbox and an irrelevant question
 * are normal outcomes - so it renders as a plain notice, not a red banner.
 */
import { useState } from 'react';
import { AnswerPanel } from './AnswerPanel.jsx';

export const AskPanel = ({ result, asking, error, onAsk, indexedItems }) => {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);

  const submit = (event) => {
    event.preventDefault();
    if (question.trim().length < 3) return;
    onAsk(question.trim(), topK);
  };

  return (
    <section className="card ask">
      <div className="card-head">
        <h2>Ask your inbox</h2>
        <label className="topk">
          top-k
          <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {[3, 5, 8, 12].map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>
      </div>

      <form onSubmit={submit} className="ask-form">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={
            indexedItems === 0
              ? 'Save something first, then ask about it'
              : 'e.g. What did I save about the rollback plan?'
          }
          aria-label="Question"
        />
        <button type="submit" className="primary" disabled={asking || question.trim().length < 3}>
          {asking ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {error && (
        <p className="error" role="alert">
          {error.message}
          {error.requestId && <span className="muted"> (request {error.requestId.slice(0, 8)})</span>}
        </p>
      )}

      {asking && <p className="muted">Retrieving relevant chunks and generating an answer...</p>}

      {!asking && result && !result.answered && (
        <p className="notice">{result.answer}</p>
      )}

      {!asking && result?.answered && <AnswerPanel result={result} />}
    </section>
  );
};
