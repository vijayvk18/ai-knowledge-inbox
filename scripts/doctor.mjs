/**
 * Diagnose why a question returned nothing.
 *
 *   npm run doctor                     # what is configured and what is indexed
 *   npm run doctor -- "your question"  # plus the score of every chunk against it
 *
 * The scores are the whole point: "nothing matches" is a *result*, and this
 * shows the numbers behind it - what was retrieved, what sat below the
 * relevance floor, and whether anything shares wording with the question at all.
 */
const BASE = process.env.API_BASE ?? 'http://localhost:8787';
const question = process.argv.slice(2).join(' ').trim();

const get = async (path) => {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(`GET ${path} -> ${response.status}`);
  return response.json();
};

const bar = (score, width = 24) => {
  const filled = Math.max(0, Math.min(width, Math.round(score * width)));
  return '█'.repeat(filled) + '·'.repeat(width - filled);
};

try {
  const health = await get('/api/health');
  const { embeddings, answers } = health.providers;

  console.log('\n=== configuration ===');
  console.log(`  embeddings : ${embeddings.provider} / ${embeddings.model} (${embeddings.dimensions ?? '?'} dims)`);
  console.log(`  answers    : ${answers.provider} / ${answers.model}`);
  if (embeddings.provider === 'local') {
    console.log('  NOTE: lexical matching only - it matches wording, not meaning.');
  }

  console.log('\n=== index ===');
  console.log(`  ${health.index.chunks} chunks from ${health.index.items} items`);
  console.log(`  item status: ${JSON.stringify(health.items)}`);
  if (health.index.chunks === 0) {
    console.log('  -> nothing is indexed. Every question will return nothing.');
  }

  const { items } = await get('/api/items?limit=50');
  console.log('\n=== saved items ===');
  for (const item of items) {
    console.log(
      `  [${item.status}] ${item.chunkCount} chunk(s), ${item.charCount} chars - ${item.title ?? '(untitled)'}`,
    );
    if (item.error) console.log(`      error: ${item.error.code} - ${item.error.message}`);
    if (item.status === 'ready' && item.charCount === 0) {
      console.log('      -> indexed but empty: nothing was extracted from this source.');
    }
    if (item.preview) console.log(`      "${item.preview.replace(/\s+/g, ' ').slice(0, 90)}..."`);
  }

  if (!question) {
    console.log('\nPass a question to score it:  npm run doctor -- "your question here"\n');
    process.exit(0);
  }

  // minScore=0 deliberately bypasses the relevance floor, so chunks that were
  // filtered out still show up with their real score.
  const unfiltered = await get(`/api/search?q=${encodeURIComponent(question)}&topK=10&minScore=0`);
  const floor = (await get(`/api/search?q=${encodeURIComponent(question)}&topK=10`)).stats.minScore;

  console.log(`\n=== scores for "${question}" ===`);
  console.log(`  relevance floor: ${floor}\n`);

  for (const result of unfiltered.results) {
    const verdict = result.score >= floor ? 'RETRIEVED' : 'below floor';
    console.log(`  ${bar(result.score)} ${result.score.toFixed(4)}  ${verdict.padEnd(11)} ${result.title ?? '(untitled)'}`);
  }

  // A score of exactly 0 is the interesting case: it is not a weak match, it is
  // no shared vocabulary at all, and that has a specific cause and a specific fix.
  const best = unfiltered.results[0];
  const bestScore = best?.score ?? 0;

  console.log('');
  if (bestScore >= floor) {
    console.log('  -> retrieval is working; this question should be answerable.');
  } else if (bestScore > 0) {
    console.log('  -> weak matches only: answered at low confidence, with a warning in the UI.');
  } else {
    console.log('  -> NOT ONE CHUNK SHARES A WORD WITH THIS QUESTION.');
    console.log('     This is why you see "nothing matches that question at all".');
    if (embeddings.provider === 'local') {
      console.log('');
      console.log('     The keyless embedder matches words, not meaning. It finds');
      console.log('     "budget cap" from "budget caps", but never "invoice deadline"');
      console.log('     from "when is payment due".');
      console.log('');
      console.log('     Two options:');
      console.log('       1. Ask using words that appear in the item above.');
      console.log('       2. Set OPENAI_API_KEY in api/.env and restart - items');
      console.log('          re-embed automatically, and paraphrases start working.');
    } else {
      console.log('     Semantic embeddings are live, so the saved content genuinely');
      console.log('     does not cover this topic.');
    }
  }
  console.log('');
} catch (error) {
  console.error(`\n  Could not reach the API at ${BASE}`);
  console.error(`  ${error.message}`);
  console.error('  Is it running? Start it with: npm run dev\n');
  process.exit(1);
}
