import { createSearch } from "./search.mjs";

const form = document.querySelector("#search-form");
const input = document.querySelector("#query");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const buttons = document.querySelectorAll(".suggestions button");

function sourceCard(passage, sourceRoot) {
  const item = document.createElement("li");
  const label = document.createElement("p");
  label.className = "result-label";
  label.textContent = passage.title;
  const heading = document.createElement("h3");
  heading.textContent = passage.heading;
  const excerpt = document.createElement("p");
  excerpt.className = "excerpt";
  excerpt.textContent = passage.text;
  const source = document.createElement("a");
  source.href = `${sourceRoot}/${passage.path}#L${passage.start}-L${passage.end}`;
  source.textContent = "View pinned source ↗";
  source.setAttribute("aria-label", `View pinned source for ${passage.heading}`);
  const notice = document.createElement("small");
  notice.textContent = passage.copyright;
  item.append(label, heading, excerpt, source, notice);
  return item;
}

try {
  const response = await fetch(new URL("passages.json", import.meta.url));
  if (!response.ok) throw new Error(`Index request returned ${response.status}`);
  const { repository, revision, passages } = await response.json();
  const sourceRoot = `${repository}/blob/${revision}`;
  document.querySelector("#revision").href = `${repository}/tree/${revision}/docs/source/en`;
  const search = createSearch(passages);

  function showResults() {
    const query = input.value.trim();
    const matches = query ? search(query) : [];
    results.replaceChildren(...matches.map((passage) => sourceCard(passage, sourceRoot)));
    if (!query) {
      status.textContent = "Enter a topic or API name.";
    } else if (matches.length === 0) {
      status.textContent = "No keyword matches in this sample. Try a different term.";
    } else {
      status.textContent = `${matches.length} ranked ${matches.length === 1 ? "excerpt" : "excerpts"}`;
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    showResults();
  });
  for (const button of buttons) {
    button.addEventListener("click", () => {
      input.value = button.textContent;
      showResults();
    });
    button.disabled = false;
  }
  document.querySelector("#submit").disabled = false;
  const pageCount = new Set(passages.map(({ path }) => path)).size;
  status.textContent = `Ready to search ${pageCount} documentation pages.`;
} catch (error) {
  console.error(error);
  status.textContent = "The browser index could not load. Reload the page to try again.";
}
