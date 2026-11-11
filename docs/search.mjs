const STOP_WORDS = new Set(
  "a an and are can do does for from how i in is it of on or that the this to use what when where which with".split(" "),
);

function terms(text) {
  return text.normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
}

export function createSearch(passages) {
  const documentFrequency = new Map();
  const documents = passages.map((passage, position) => {
    const tokens = terms(`${passage.heading} ${passage.text}`);
    const counts = new Map();
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
    for (const token of counts.keys()) {
      documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
    }
    return { passage, position, length: tokens.length, counts };
  });
  const meanLength = documents.reduce((sum, document) => sum + document.length, 0) / documents.length || 1;
  const k1 = 1.2;
  const b = 0.75;

  return function search(query, limit = 3) {
    const queryTerms = [...new Set(terms(query))].filter((term) => !STOP_WORDS.has(term));
    const matches = [];
    for (const document of documents) {
      const lengthFactor = 1 - b + b * document.length / meanLength;
      let score = 0;
      for (const term of queryTerms) {
        const count = document.counts.get(term);
        if (!count) continue;
        const frequency = documentFrequency.get(term);
        const idf = Math.log1p((documents.length - frequency + 0.5) / (frequency + 0.5));
        score += idf * count * (k1 + 1) / (count + k1 * lengthFactor);
      }
      if (score > 0) matches.push({ ...document, score });
    }

    return matches
      .sort((first, second) => second.score - first.score || first.position - second.position)
      .slice(0, limit)
      .map(({ passage }) => passage);
  };
}
