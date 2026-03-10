import { useState, useEffect } from "react";
import { AlertCircle, Camera, Clock, MapPin, AlertTriangle } from "lucide-react";
import { Badge } from "./ui/badge";

interface Detection {
  id: string;
  type: string;
  location: string;
  camera: string;
  timestamp: Date;
  severity: "low" | "medium" | "high";
  confidence: number;
}

export function LiveFeed() {
  const [detections, setDetections] = useState<Detection[]>([
    {
      id: "1",
      type: "Fighting",
      location: "Hallway B - 2nd Floor",
      camera: "CAM-204",
      timestamp: new Date(),
      severity: "high",
      confidence: 94,
    },
    {
      id: "2",
      type: "Running in hallway",
      location: "Main Corridor",
      camera: "CAM-101",
      timestamp: new Date(Date.now() - 120000),
      severity: "low",
      confidence: 87,
    },
    {
      id: "3",
      type: "Unauthorized area",
      location: "Staff Room",
      camera: "CAM-305",
      timestamp: new Date(Date.now() - 300000),
      severity: "medium",
      confidence: 91,
    },
  ]);

  const [activeCamera, setActiveCamera] = useState("CAM-204");

  useEffect(() => {
    // Simulate new detections
    const interval = setInterval(() => {
      const types = ["Running in hallway", "Littering", "Loud behavior", "Phone usage"];
      const locations = ["Hallway A", "Cafeteria", "Library", "Gymnasium"];
      const cameras = ["CAM-101", "CAM-204", "CAM-305", "CAM-412"];
      const severities: ("low" | "medium" | "high")[] = ["low", "medium"];

      const newDetection: Detection = {
        id: Date.now().toString(),
        type: types[Math.floor(Math.random() * types.length)],
        location: locations[Math.floor(Math.random() * locations.length)],
        camera: cameras[Math.floor(Math.random() * cameras.length)],
        timestamp: new Date(),
        severity: severities[Math.floor(Math.random() * severities.length)],
        confidence: Math.floor(Math.random() * 15) + 80,
      };

      setDetections((prev) => [newDetection, ...prev].slice(0, 10));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      case "medium":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "low":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      default:
        return "bg-zinc-500/20 text-zinc-400 border-zinc-500/30";
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleTimeString();
  };

  return (
    <div className="h-full p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-100">Live Feed Monitoring</h2>
          <p className="text-zinc-400 text-sm mt-1">Real-time misconduct detection across campus</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/30 px-3 py-1">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
            12 Cameras Active
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100%-5rem)]">
        {/* Main Video Feed */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden h-full flex flex-col">
            {/* Video Display */}
            <div className="relative bg-zinc-950 aspect-video flex-shrink-0">
              {/* Mock video feed */}
              <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 via-zinc-900 to-zinc-950">
                <div className="absolute inset-0 flex items-center justify-center">
                  <Camera className="w-16 h-16 text-zinc-700" />
                </div>
                
                {/* Detection overlay */}
                <div className="absolute top-1/3 left-1/4 w-32 h-32 border-2 border-red-500 animate-pulse">
                  <div className="absolute -top-8 left-0 bg-red-500 text-white px-2 py-1 rounded text-sm font-medium">
                    Fighting Detected
                  </div>
                </div>

                {/* Camera info overlay */}
                <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
                  <div className="flex items-center gap-2 text-white">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                    <span className="text-sm font-medium">{activeCamera}</span>
                  </div>
                </div>

                {/* Timestamp overlay */}
                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
                  <span className="text-white text-sm font-mono">
                    {new Date().toLocaleTimeString()}
                  </span>
                </div>

                {/* Confidence indicator */}
                <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
                  <div className="text-white text-sm">
                    <span className="text-zinc-400">Confidence:</span>{" "}
                    <span className="font-medium">94%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Camera Grid */}
            <div className="p-4 grid grid-cols-4 gap-3">
              {["CAM-101", "CAM-204", "CAM-305", "CAM-412"].map((cam) => (
                <button
                  key={cam}
                  onClick={() => setActiveCamera(cam)}
                  className={`aspect-video rounded-lg border-2 transition-all duration-200 ${
                    activeCamera === cam
                      ? "border-blue-500 bg-blue-500/10"
                      : "border-zinc-700 bg-zinc-800 hover:border-zinc-600"
                  }`}
                >
                  <div className="flex flex-col items-center justify-center h-full">
                    <Camera className="w-5 h-5 text-zinc-500 mb-1" />
                    <span className="text-xs text-zinc-400">{cam}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Detections */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 overflow-hidden flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            <h3 className="font-semibold text-zinc-100">Recent Detections</h3>
          </div>

          <div className="space-y-3 overflow-y-auto flex-1 pr-2">
            {detections.map((detection) => (
              <div
                key={detection.id}
                className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3 hover:bg-zinc-800 transition-colors cursor-pointer"
              >
                <div className="flex items-start justify-between mb-2">
                  <Badge
                    variant="outline"
                    className={`text-xs font-medium ${getSeverityColor(detection.severity)}`}
                  >
                    {detection.severity.toUpperCase()}
                  </Badge>
                  <span className="text-xs text-zinc-500">{formatTime(detection.timestamp)}</span>
                </div>

                <h4 className="font-medium text-zinc-100 mb-2">{detection.type}</h4>

                <div className="space-y-1 text-xs text-zinc-400">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-3 h-3" />
                    <span>{detection.location}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Camera className="w-3 h-3" />
                    <span>{detection.camera}</span>
                  </div>
                  <div className="flex items-center justify-between mt-2 pt-2 border-t border-zinc-700">
                    <span>Confidence</span>
                    <span className="font-medium text-zinc-300">{detection.confidence}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
