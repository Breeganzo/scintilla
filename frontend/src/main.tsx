import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./index.css";

const root = document.getElementById("root");
if (!root) {
  // Not defensive noise: if index.html ever loses the mount point, an explicit
  // error is far easier to diagnose than a silently blank page.
  throw new Error("#root is missing from index.html");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
