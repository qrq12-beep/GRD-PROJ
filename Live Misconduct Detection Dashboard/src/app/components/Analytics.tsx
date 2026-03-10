import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle2, Clock, Camera } from "lucide-react";
import { Card } from "./ui/card";

export function Analytics() {
  const weeklyData = [
    { day: "Mon", incidents: 12, resolved: 10 },
    { day: "Tue", incidents: 19, resolved: 16 },
    { day: "Wed", incidents: 8, resolved: 8 },
    { day: "Thu", incidents: 15, resolved: 13 },
    { day: "Fri", incidents: 22, resolved: 18 },
    { day: "Sat", incidents: 5, resolved: 5 },
    { day: "Sun", incidents: 3, resolved: 3 },
  ];

  const incidentTypes = [
    { name: "Fighting", value: 15, color: "#ef4444" },
    { name: "Running", value: 28, color: "#f97316" },
    { name: "Unauthorized Area", value: 22, color: "#eab308" },
    { name: "Littering", value: 18, color: "#3b82f6" },
    { name: "Loud Behavior", value: 17, color: "#8b5cf6" },
  ];

  const hourlyTrend = [
    { hour: "8AM", count: 5 },
    { hour: "9AM", count: 8 },
    { hour: "10AM", count: 12 },
    { hour: "11AM", count: 15 },
    { hour: "12PM", count: 22 },
    { hour: "1PM", count: 18 },
    { hour: "2PM", count: 14 },
    { hour: "3PM", count: 20 },
    { hour: "4PM", count: 10 },
  ];

  const stats = [
    {
      title: "Total Incidents",
      value: "84",
      change: "+12%",
      trend: "up",
      icon: AlertTriangle,
      color: "text-orange-400",
      bgColor: "bg-orange-500/10",
    },
    {
      title: "Resolved",
      value: "73",
      change: "+8%",
      trend: "up",
      icon: CheckCircle2,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
    },
    {
      title: "Avg Response Time",
      value: "2.4m",
      change: "-15%",
      trend: "down",
      icon: Clock,
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
    },
    {
      title: "Active Cameras",
      value: "12",
      change: "100%",
      trend: "neutral",
      icon: Camera,
      color: "text-purple-400",
      bgColor: "bg-purple-500/10",
    },
  ];

  return (
    <div className="h-full p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold text-zinc-100">Analytics Dashboard</h2>
        <p className="text-zinc-400 text-sm mt-1">Insights and trends from detected misconducts</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="bg-zinc-900 border-zinc-800 p-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <p className="text-zinc-400 text-sm">{stat.title}</p>
                <p className="text-2xl font-semibold text-zinc-100 mt-2">{stat.value}</p>
                <div className="flex items-center gap-1 mt-2">
                  {stat.trend === "up" && <TrendingUp className="w-4 h-4 text-green-400" />}
                  {stat.trend === "down" && <TrendingDown className="w-4 h-4 text-green-400" />}
                  <span className={`text-sm ${stat.trend === "neutral" ? "text-zinc-400" : "text-green-400"}`}>
                    {stat.change}
                  </span>
                  <span className="text-sm text-zinc-500">vs last week</span>
                </div>
              </div>
              <div className={`${stat.bgColor} ${stat.color} p-3 rounded-lg`}>
                <stat.icon className="w-6 h-6" />
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Incidents */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">Weekly Incidents</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="day" stroke="#71717a" />
              <YAxis stroke="#71717a" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #27272a",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Legend />
              <Bar dataKey="incidents" fill="#3b82f6" radius={[8, 8, 0, 0]} />
              <Bar dataKey="resolved" fill="#10b981" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Incident Types */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">Incident Types Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={incidentTypes}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {incidentTypes.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #27272a",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Hourly Trend */}
        <Card className="bg-zinc-900 border-zinc-800 p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">Hourly Activity Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={hourlyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="hour" stroke="#71717a" />
              <YAxis stroke="#71717a" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#18181b",
                  border: "1px solid #27272a",
                  borderRadius: "8px",
                  color: "#fff",
                }}
              />
              <Line
                type="monotone"
                dataKey="count"
                stroke="#8b5cf6"
                strokeWidth={3}
                dot={{ fill: "#8b5cf6", r: 5 }}
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Recent Alerts */}
      <Card className="bg-zinc-900 border-zinc-800 p-6">
        <h3 className="text-lg font-semibold text-zinc-100 mb-4">Top Locations</h3>
        <div className="space-y-3">
          {[
            { location: "Hallway B - 2nd Floor", incidents: 18, percentage: 85 },
            { location: "Main Corridor", incidents: 15, percentage: 70 },
            { location: "Cafeteria", incidents: 12, percentage: 55 },
            { location: "Gymnasium", incidents: 10, percentage: 45 },
            { location: "Library", incidents: 8, percentage: 35 },
          ].map((item) => (
            <div key={item.location} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-zinc-300">{item.location}</span>
                <span className="text-zinc-400">{item.incidents} incidents</span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${item.percentage}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
