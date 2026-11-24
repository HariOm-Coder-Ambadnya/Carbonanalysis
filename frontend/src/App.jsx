
import { useState } from "react";
import {
  LayoutDashboard,
  Activity,
  BrainCircuit,
  CalendarClock,
  ChartNoAxesCombined,
  BookOpen,
  Menu,
  Sun,
  Moon,
  Leaf,
  Zap,
  Cloud,
  DollarSign,
} from "lucide-react";

import Performance from "./pages/Performance";
import Recommendations from "./pages/Recommendations";

const navigation = [
  { name: "Dashboard", icon: LayoutDashboard },
  { name: "Workload Analysis", icon: Activity },
  { name: "ML Recommendations", icon: BrainCircuit },
  { name: "Scheduling", icon: CalendarClock },
  { name: "Performance", icon: ChartNoAxesCombined },
  { name: "Methodology", icon: BookOpen },
];

const metrics = [
  {
    label: "Average Energy",
    value: "0.415 kWh",
    note: "Multi-objective strategy",
    icon: Zap,
  },
  {
    label: "Carbon Emissions",
    value: "289.00 g",
    note: "Estimated CO₂e",
    icon: Leaf,
  },
  {
    label: "Average Cost",
    value: "$1.93",
    note: "Estimated gross cost",
    icon: DollarSign,
  },
  {
    label: "Scenarios Evaluated",
    value: "56",
    note: "Simulation scenarios",
    icon: Cloud,
  },
];

function Dashboard() {
  return (
    <>
      <div className="section-heading">
        <h2>Performance Overview</h2>
        <span>Historical simulation averages</span>
      </div>

      <div className="metrics-grid">
        {metrics.map(({ label, value, note, icon: Icon }) => (
          <article className="metric-card" key={label}>
            <div className="metric-top">
              <span>{label}</span>
              <div className="metric-icon">
                <Icon size={20} />
              </div>
            </div>
            <h2>{value}</h2>
            <p>{note}</p>
          </article>
        ))}
      </div>

      <div className="section-heading">
        <h2>Project Summary</h2>
        <span>Evaluation status</span>
      </div>

      <div className="summary-grid">
        <article className="panel">
          <h3>Evaluation coverage</h3>
          <div className="big-number">840</div>
          <p>Successful strategy and deadline evaluations</p>
          <div className="progress-track">
            <div className="progress-fill" />
          </div>
          <small>56 scenarios · 5 strategies · 3 deadlines</small>
        </article>

        <article className="panel">
          <h3>Available strategies</h3>

          {[
            "Baseline",
            "Carbon Aware",
            "Approximate Capacity",
            "Exact Capacity",
            "Multi Objective",
          ].map((item) => (
            <div className="strategy-row" key={item}>
              <span className="strategy-marker" />
              <span>{item}</span>
              <small>168 runs</small>
            </div>
          ))}
        </article>
      </div>

      <div className="notice">
        <strong>Important:</strong> Energy and carbon values are simulation
        estimates, not physical measurements. Cost is unchanged across
        strategies in the current model.
      </div>
    </>
  );
}

function ComingSoon({ active }) {
  return (
    <div className="panel coming-soon">
      <div className="metric-icon">
        <BrainCircuit size={25} />
      </div>

      <h2>{active}</h2>

      <p>
        This module is next in our implementation. We'll connect it to
        the verified project data and backend.
      </p>
    </div>
  );
}

function App() {
  const [dark, setDark] = useState(true);
  const [active, setActive] = useState("Dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const renderPage = () => {
    switch (active) {
      case "Dashboard":
        return <Dashboard />;

      case "ML Recommendations":
        return <Recommendations />;

      case "Performance":
        return <Performance />;

      default:
        return <ComingSoon active={active} />;
    }
  };

  return (
    <div className={dark ? "app dark" : "app light"}>
      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="brand">
          <div className="brand-icon">
            <Leaf size={22} />
          </div>

          {sidebarOpen && (
            <div>
              <strong>CarbonControl</strong>
              <small>Serverless Intelligence</small>
            </div>
          )}
        </div>

        <p className="nav-label">WORKSPACE</p>

        <nav>
          {navigation.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${
                active === name ? "selected" : ""
              }`}
              onClick={() => setActive(name)}
              title={name}
            >
              <Icon size={19} />

              {sidebarOpen && <span>{name}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />

          {sidebarOpen && "Simulation environment"}
        </div>
      </aside>

      {/* Main content */}
      <main className="main">
        {/* Top navigation */}
        <header className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            <Menu size={20} />
          </button>

          <div className="breadcrumb">
            Workspace / <strong>{active}</strong>
          </div>

          <button
            className="icon-button theme-button"
            onClick={() => setDark(!dark)}
            aria-label="Toggle theme"
          >
            {dark ? <Sun size={19} /> : <Moon size={19} />}
          </button>
        </header>

        {/* Page content */}
        <section className="content">
          {active !== "ML Recommendations" && (
            <div className="page-heading">
              <div>
                <span className="eyebrow">
                  SERVERLESS ANALYTICS PLATFORM
                </span>

                <h1>{active}</h1>

                <p>
                  Monitor workload scheduling, energy consumption and
                  carbon impact.
                </p>
              </div>

              <div className="environment-tag">
                <span className="status-dot" />
                Offline simulation
              </div>
            </div>
          )}

          {renderPage()}
        </section>
      </main>
    </div>
  );
}

export default App;
