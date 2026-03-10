import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { LiveFeed } from "./components/LiveFeed";
import { Analytics } from "./components/Analytics";
import { Settings } from "./components/Settings";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: LiveFeed },
      { path: "analytics", Component: Analytics },
      { path: "settings", Component: Settings },
    ],
  },
]);
