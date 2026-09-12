/**
 * The answer and the evidence behind it.
 *
 * Citations are the whole point of RAG output, so [n] markers in the answer are
 * rendered as buttons that jump to the matching source card. A retrieved
 * passage the model did not cite is shown dimmed rather than hidden - it is
 * honest about what was in context, and it is the first thing you want when
 * tuning retrieval.
 */
import { useState } from 'react';

const SNIPPET_CLAMP = 320;

/** Split answer text into plain runs and [n] citation markers. */
const segments = (text) => {
  const parts = [];
  let lastIndex = 0;

  for (const match of text.matchAll(/\[(\d{1,2})\]/g)) {
    if (match.index > lastIndex) parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    parts.push({ type: 'citation', value: Number(match[1]) });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) parts.push({ type: 'text', value: text.slice(lastIndex) });
  return parts;
};

const AnswerText = ({ text, onJumpTo }) => (
  <p className="answer-text">
    {segments(text).map((segment, index) =>
      segment.type === 'text' ? (
        <span key={index}>{segment.value}</span>
      ) : (
        <button
          key={index}
          type="button"
          className="citation"
          onClick={() => onJumpTo(segment.value)}
          title={`Jump to source ${segment.value}`}
        >
          {segment.value}
        </button>
      ),
    )}
  </p>
);

const SourceCard = ({ source }) => {
  const [expanded, setExpanded] = useState(false);
  const isLong = source.snippet.length > SNIPPET_CLAMP;
  const shown = expanded || !isLong ? source.snippet : `${source.snippet.slice(0, SNIPPET_CLAMP)}...`;

  return (
    <li id={`source-${source.citation}`} className={source.cited ? 'source' : 'source source-uncited'}>
      <div className="source-head">
        <span className="source-number">{source.citation}</span>
        <div>
          <h4 className="source-title">{source.title || 'Untitled note'}</h4>
          <span className="muted source-meta">
            {source.sourceType}
            {' - '}
            score {source.score.toFixed(3)}
            {!source.cited && ' - retrieved but not cited'}
          </span>
        </div>
      </div>

      {source.url && (
        <a className="item-url" href={source.url} target="_blank" rel="noreferrer noopener">
          {source.url}
        </a>
      )}

      <p className="source-snippet">{shown}</p>
      {isLong && (
        <button type="button" className="link" onClick={() => setExpanded((value) => !value)}>
          {expanded ? 'Show less' : 'Show full chunk'}
        </button>
      )}
    </li>
  );
};

export const AnswerPanel = ({ result }) => {
  const jumpTo = (citation) => {
    const element = document.getElementById(`source-${citation}`);
    element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    element?.classList.add('source-flash');
    setTimeout(() => element?.classList.remove('source-flash'), 1200);
  };

  return (
    <section className="answer">
      {/* Nothing cleared the relevance floor, so these are best-effort matches.
          Say so up front rather than presenting a weak match as a firm answer. */}
      {result.lowConfidence && (
        <p className="notice">
          Weak match - nothing in your saved items scored as clearly relevant, so this is the
          closest thing found. Check the sources below before trusting it.
        </p>
      )}

      <AnswerText text={result.answer} onJumpTo={jumpTo} />

      {result.truncated && (
        <p className="muted">The answer hit the output limit and may be cut off.</p>
      )}

      {result.sources.length > 0 && (
        <>
          <h3 className="answer-sources-heading">Sources</h3>
          <ul className="source-list">
            {result.sources.map((source) => (
              <SourceCard key={source.chunkId} source={source} />
            ))}
          </ul>
        </>
      )}

      {/* Retrieval telemetry, inline. Tuning topK or a relevance floor blind is
          guesswork; these are the numbers you actually need. */}
      <details className="stats">
        <summary>Pipeline details</summary>
        <dl>
          <div><dt>Retrieved</dt><dd>{result.stats.retrievedChunks} of {result.stats.indexedChunks} chunks</dd></div>
          <div><dt>Embedding</dt><dd>{result.stats.embeddingProvider} / {result.stats.embeddingModel}</dd></div>
          <div><dt>Answer</dt><dd>{result.stats.answerProvider} / {result.stats.answerModel ?? 'n/a'}</dd></div>
          <div><dt>Relevance floor</dt><dd>{result.stats.minScore}</dd></div>
          <div><dt>Latency</dt><dd>
            {result.stats.totalMs} ms total ({result.stats.embeddingMs ?? 0} embed, {result.stats.searchMs ?? 0} search, {result.stats.generationMs ?? 0} generate)
          </dd></div>
          {result.stats.inputTokens > 0 && (
            <div><dt>Tokens</dt><dd>{result.stats.inputTokens} in / {result.stats.outputTokens} out</dd></div>
          )}
        </dl>
      </details>
    </section>
  );
};
