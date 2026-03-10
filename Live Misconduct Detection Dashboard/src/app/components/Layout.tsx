import { Outlet, NavLink } from "react-router";
import { Video, BarChart3, Settings, Shield } from "lucide-react";

export function Layout() {
  const navItems = [
    { to: "/", icon: Video, label: "Live Feed", exact: true },
    { to: "/analytics", icon: BarChart3, label: "Analytics", exact: false },
    { to: "/settings", icon: Settings, label: "Settings", exact: false },
  ];

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      {/* Sidebar */}
      <aside className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-lg">SafeSchool AI</h1>
              <p className="text-xs text-zinc-500">Misconduct Detection</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.exact}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-600/30"
                    : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                }`
              }
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-zinc-800">
          <div className="px-4 py-3 bg-zinc-800/50 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium">System Active</span>
            </div>
            <p className="text-xs text-zinc-500">All cameras online</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
