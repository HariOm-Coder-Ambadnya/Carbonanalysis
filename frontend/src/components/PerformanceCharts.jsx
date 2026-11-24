
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const colors = {
  energy: "#38bdf8",
  carbon: "#38c99a",
  cost: "#f5b942",
};

function Chart({ title, data, dataKey, color, unit }) {
  return (
    <article className="panel chart-panel">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 45 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="strategy"
            angle={-20}
            textAnchor="end"
            interval={0}
            tick={{ fill: "var(--muted)", fontSize: 10 }}
          />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(4)} ${unit}`, title]}
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text)",
            }}
          />
          <Bar dataKey={dataKey} fill={color} radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </article>
  );
}

export default function PerformanceCharts({ data }) {
  return (
    <div className="charts-grid">
      <Chart title="Average Energy Consumption" data={data}
        dataKey="energy" color={colors.energy} unit="kWh" />
      <Chart title="Average Carbon Emissions" data={data}
        dataKey="carbon" color={colors.carbon} unit="gCO₂e" />
      <Chart title="Average Execution Cost" data={data}
        dataKey="cost" color={colors.cost} unit="USD" />
    </div>
  );
}
