
import { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { performanceData } from "../data/performanceData";

const charts = [
  { key: "energy", title: "Energy Consumption", unit: "kWh", color: "#38bdf8" },
  { key: "carbon", title: "Carbon Emissions", unit: "gCO₂e", color: "#38c99a" },
  { key: "cost", title: "Execution Cost", unit: "USD", color: "#f5b942" },
];

function MetricChart({ config }) {
  return (
    <article className="panel chart-panel">
      <h3>{config.title}</h3>
      <ResponsiveContainer width="100%" height={290}>
        <BarChart
          data={performanceData}
          margin={{ top: 15, right: 10, left: 5, bottom: 45 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="strategy"
            interval={0}
            angle={-20}
            textAnchor="end"
            tick={{ fill: "var(--muted)", fontSize: 10 }}
          />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} />
          <Tooltip
            formatter={(value) => [
              `${Number(value).toFixed(6)} ${config.unit}`,
              config.title,
            ]}
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text)",
            }}
          />
          <Bar dataKey={config.key} fill={config.color} radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </article>
  );
}

export default function Performance() {
  const [selected, setSelected] = useState("ALL");

  const filteredData =
    selected === "ALL"
      ? performanceData
      : performanceData.filter((item) => item.strategy === selected);

  return (
    <div>
      <div className="section-heading">
        <h2>Strategy Performance</h2>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="ALL">All strategies</option>
          {performanceData.map((item) => (
            <option key={item.strategy} value={item.strategy}>
              {item.strategy}
            </option>
          ))}
        </select>
      </div>

      <div className="charts-grid">
        {charts.map((chart) => (
          <MetricChart key={chart.key} config={chart} />
        ))}
      </div>

      <div className="panel performance-table">
        <h3>Evaluation Summary</h3>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Energy (kWh)</th>
                <th>Carbon (gCO₂e)</th>
                <th>Cost (USD)</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((item) => (
                <tr key={item.strategy}>
                  <td>{item.strategy}</td>
                  <td>{item.energy.toFixed(6)}</td>
                  <td>{item.carbon.toFixed(6)}</td>
                  <td>{item.cost.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="notice">
        <strong>Evaluation note:</strong> These are averages across 168 successful
        evaluations per strategy. Energy and carbon are simulation estimates.
        Gross cost is identical across all five strategies in the current model.
      </div>
    </div>
  );
}
