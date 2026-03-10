import { useState } from "react";
import { Bell, Camera, Shield, Sliders, Save, Users, Zap } from "lucide-react";
import { Card } from "./ui/card";
import { Switch } from "./ui/switch";
import { Slider } from "./ui/slider";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Separator } from "./ui/separator";
import { toast } from "sonner";

export function Settings() {
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [soundAlerts, setSoundAlerts] = useState(false);
  const [autoArchive, setAutoArchive] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState([75]);
  const [recordingQuality, setRecordingQuality] = useState([80]);

  const handleSave = () => {
    toast.success("Settings saved successfully!");
  };

  return (
    <div className="h-full p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-zinc-100">Settings</h2>
          <p className="text-zinc-400 text-sm mt-1">Configure your detection system preferences</p>
        </div>
        <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700">
          <Save className="w-4 h-4 mr-2" />
          Save Changes
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Detection Settings */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-purple-500/10 rounded-lg">
              <Sliders className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-zinc-100">Detection Settings</h3>
              <p className="text-sm text-zinc-500">Configure AI detection parameters</p>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-3">
                <Label htmlFor="confidence" className="text-zinc-300">
                  Confidence Threshold
                </Label>
                <span className="text-sm text-zinc-400">{confidenceThreshold[0]}%</span>
              </div>
              <Slider
                id="confidence"
                value={confidenceThreshold}
                onValueChange={setConfidenceThreshold}
                max={100}
                step={5}
                className="w-full"
              />
              <p className="text-xs text-zinc-500 mt-2">
                Minimum confidence level required to trigger an alert
              </p>
            </div>

            <Separator className="bg-zinc-800" />

            <div className="space-y-4">
              <h4 className="text-sm font-medium text-zinc-300">Detection Types</h4>
              {[
                { label: "Fighting Detection", checked: true },
                { label: "Running Detection", checked: true },
                { label: "Unauthorized Area Access", checked: true },
                { label: "Crowd Detection", checked: false },
                { label: "Weapon Detection", checked: true },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <Label className="text-zinc-400">{item.label}</Label>
                  <Switch defaultChecked={item.checked} />
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Notification Settings */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Bell className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-zinc-100">Notifications</h3>
              <p className="text-sm text-zinc-500">Manage alert preferences</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-zinc-300">Email Notifications</Label>
                <p className="text-xs text-zinc-500 mt-1">Receive alerts via email</p>
              </div>
              <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
            </div>

            <Separator className="bg-zinc-800" />

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-zinc-300">Push Notifications</Label>
                <p className="text-xs text-zinc-500 mt-1">Browser push notifications</p>
              </div>
              <Switch checked={pushNotifications} onCheckedChange={setPushNotifications} />
            </div>

            <Separator className="bg-zinc-800" />

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-zinc-300">Sound Alerts</Label>
                <p className="text-xs text-zinc-500 mt-1">Play sound on detection</p>
              </div>
              <Switch checked={soundAlerts} onCheckedChange={setSoundAlerts} />
            </div>

            <Separator className="bg-zinc-800" />

            <div>
              <Label htmlFor="email" className="text-zinc-300">
                Notification Email
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="admin@school.edu"
                className="mt-2 bg-zinc-800 border-zinc-700 text-zinc-100"
              />
            </div>
          </div>
        </Card>

        {/* Camera Settings */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Camera className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-zinc-100">Camera Configuration</h3>
              <p className="text-sm text-zinc-500">Manage camera settings</p>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-3">
                <Label htmlFor="quality" className="text-zinc-300">
                  Recording Quality
                </Label>
                <span className="text-sm text-zinc-400">{recordingQuality[0]}%</span>
              </div>
              <Slider
                id="quality"
                value={recordingQuality}
                onValueChange={setRecordingQuality}
                max={100}
                step={10}
                className="w-full"
              />
              <p className="text-xs text-zinc-500 mt-2">
                Higher quality uses more storage space
              </p>
            </div>

            <Separator className="bg-zinc-800" />

            <div className="space-y-3">
              <h4 className="text-sm font-medium text-zinc-300">Active Cameras</h4>
              {["CAM-101 - Main Corridor", "CAM-204 - Hallway B", "CAM-305 - Staff Room", "CAM-412 - Gymnasium"].map(
                (cam) => (
                  <div
                    key={cam}
                    className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-green-500 rounded-full" />
                      <span className="text-sm text-zinc-300">{cam}</span>
                    </div>
                    <Button variant="outline" size="sm" className="text-xs">
                      Configure
                    </Button>
                  </div>
                )
              )}
            </div>
          </div>
        </Card>

        {/* System Settings */}
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-orange-500/10 rounded-lg">
              <Zap className="w-5 h-5 text-orange-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-zinc-100">System Preferences</h3>
              <p className="text-sm text-zinc-500">General system settings</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-zinc-300">Auto-Archive Old Incidents</Label>
                <p className="text-xs text-zinc-500 mt-1">Archive incidents after 30 days</p>
              </div>
              <Switch checked={autoArchive} onCheckedChange={setAutoArchive} />
            </div>

            <Separator className="bg-zinc-800" />

            <div>
              <Label htmlFor="retention" className="text-zinc-300">
                Data Retention Period
              </Label>
              <Input
                id="retention"
                type="number"
                placeholder="90"
                className="mt-2 bg-zinc-800 border-zinc-700 text-zinc-100"
              />
              <p className="text-xs text-zinc-500 mt-2">Number of days to keep recordings</p>
            </div>

            <Separator className="bg-zinc-800" />

            <div>
              <Label htmlFor="timezone" className="text-zinc-300">
                Time Zone
              </Label>
              <Input
                id="timezone"
                type="text"
                placeholder="UTC-5 (Eastern Time)"
                className="mt-2 bg-zinc-800 border-zinc-700 text-zinc-100"
              />
            </div>
          </div>
        </Card>
      </div>

      {/* User Management Section */}
      <Card className="bg-zinc-900 border-zinc-800 p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-red-500/10 rounded-lg">
            <Users className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-zinc-100">User Access</h3>
            <p className="text-sm text-zinc-500">Manage staff access permissions</p>
          </div>
        </div>

        <div className="space-y-3">
          {[
            { name: "John Smith", role: "Administrator", email: "j.smith@school.edu" },
            { name: "Sarah Johnson", role: "Security Staff", email: "s.johnson@school.edu" },
            { name: "Mike Davis", role: "Viewer", email: "m.davis@school.edu" },
          ].map((user) => (
            <div
              key={user.email}
              className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg"
            >
              <div>
                <p className="font-medium text-zinc-100">{user.name}</p>
                <p className="text-sm text-zinc-400">{user.email}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-zinc-500">{user.role}</span>
                <Button variant="outline" size="sm">
                  Edit
                </Button>
              </div>
            </div>
          ))}
        </div>

        <Button variant="outline" className="w-full mt-4">
          <Users className="w-4 h-4 mr-2" />
          Add New User
        </Button>
      </Card>
    </div>
  );
}
