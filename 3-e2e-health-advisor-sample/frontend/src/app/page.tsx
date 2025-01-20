"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useTheme } from "next-themes";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { 
  Sun, Moon, TestTube, Activity, 
  Microscope, LineChart, Boxes,
  AlertTriangle, Dna, Pill, 
  Users, TrendingUp, Info,
  Loader2
} from "lucide-react";
import { cn } from "@/lib/utils";

// Hover effect styles
const hoverCardStyles = "hover:scale-105 transition-transform duration-200";

// TypeScript interfaces
interface DrugCandidate {
  id: string;
  molecule_type: string;
  therapeutic_area: string;
  predicted_efficacy: number;
  predicted_safety: number;
  development_stage: string;
}

interface ClinicalTrial {
  trial_id: string;
  phase: string;
  status: string;
  participant_count: number;
  target_participant_count: number;
  real_time_metrics: {
    enrollment_rate: number;
    retention_rate: number;
    safety_signals: string[];
  };
}

interface DrugCandidate {
  id: string;
  molecule_type: string;
  therapeutic_area: string;
  predicted_efficacy: number;
  predicted_safety: number;
  development_stage: string;
}

interface ClinicalTrial {
  trial_id: string;
  phase: string;
  status: string;
  participant_count: number;
  target_participant_count: number;
  real_time_metrics: {
    enrollment_rate: number;
    retention_rate: number;
    safety_signals: string[];
  };
}

export default function Home() {
  const [drugCandidates, setDrugCandidates] = useState<DrugCandidate[]>([]);
  const [trials, setTrials] = useState<ClinicalTrial[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [mounted, setMounted] = useState(false);
  const { theme, setTheme } = useTheme();
  const [error, setError] = useState<string | null>(null);

  // Function to handle manual refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const response = await fetch('http://localhost:8000/api/drugs');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDrugCandidates(data);
      setError(null);
    } catch (error) {
      console.error('Error refreshing data:', error);
      setError('Failed to refresh drug data');
    } finally {
      setRefreshing(false);
    }
  };

  // Prevent hydration mismatch by only rendering theme-dependent content after mount
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/drugs'); // Adjust URL to match your backend
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDrugCandidates(data);
      } catch (error) {
        console.error('Error fetching data:', error);
        // Handle error appropriately in the UI
        setError('Failed to fetch drug data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Prevent hydration mismatch by not rendering until mounted
  if (!mounted) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <nav className="flex justify-between items-center mb-8">
          <div className="flex items-center gap-2">
            <TestTube className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">
              Drug Development Platform
            </h1>
            <div className="ml-4 text-sm text-muted-foreground">
              🧪 AI-powered drug discovery and development
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing || loading}
              className="flex items-center gap-2"
            >
              {refreshing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <TrendingUp className="h-4 w-4" />
              )}
              {refreshing ? 'Refreshing...' : 'Refresh Data'}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="rounded-full"
            >
              {theme === "dark" ? (
                <Sun className="h-5 w-5 text-yellow-500 transition-all" />
              ) : (
                <Moon className="h-5 w-5 text-slate-900 transition-all" />
              )}
            </Button>
          </div>
        </nav>

        <Card className="p-6 mb-8 border-primary/20 shadow-lg bg-destructive/5">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            <h2 className="font-semibold">Research Disclaimer</h2>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            This platform is for research and development purposes only. All predictions and analyses should be validated through proper scientific methods and regulatory procedures.
          </p>
        </Card>

        <div className="grid gap-6 mb-8 grid-cols-1 lg:grid-cols-4">
          <Card className="col-span-1 lg:col-span-2 p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <CardHeader className="p-0 mb-4">
              <CardTitle className="flex items-center gap-2">
                <Dna className="h-5 w-5 text-primary" />
                Molecular Design
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                🧬 AI-generated drug candidates with predicted efficacy and safety profiles
              </p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-4">
                {loading ? (
                  <div className="flex items-center justify-center h-32">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : drugCandidates.length === 0 ? (
                  <div className="flex items-center justify-center h-32 text-muted-foreground">
                    No drug candidates available
                  </div>
                ) : (
                  drugCandidates.slice(0, 3).map((candidate) => (
                    <div key={candidate.id} className={cn("p-4 rounded-lg bg-background/50", hoverCardStyles)}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">{candidate.id}</span>
                        <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
                          {candidate.therapeutic_area}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <div className="flex items-center gap-1 cursor-help">
                                  <p className="text-muted-foreground">Efficacy</p>
                                  <Info className="h-3 w-3 text-muted-foreground" />
                                </div>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>AI-predicted effectiveness of the drug candidate</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <p className="font-medium">{(candidate.predicted_efficacy * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <div className="flex items-center gap-1 cursor-help">
                                  <p className="text-muted-foreground">Safety</p>
                                  <Info className="h-3 w-3 text-muted-foreground" />
                                </div>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>AI-predicted safety profile score</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <p className="font-medium">{(candidate.predicted_safety * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-1 lg:col-span-2 p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <CardHeader className="p-0 mb-4">
              <CardTitle className="flex items-center gap-2">
                <Microscope className="h-5 w-5 text-primary" />
                Clinical Trials
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-2">
                🔬 Real-time monitoring of ongoing clinical trials and patient responses
              </p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-4">
                {loading ? (
                  <div className="flex items-center justify-center h-32">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : trials?.length > 0 ? trials.slice(0, 3).map((trial) => (
                  <div key={trial.trial_id} className={cn("p-4 rounded-lg bg-background/50", hoverCardStyles)}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">{trial.trial_id}</span>
                      <span className={cn("text-xs px-2 py-1 rounded-full transition-colors", {
                        "bg-green-500/10 text-green-500 hover:bg-green-500/20": trial.status === "active",
                        "bg-yellow-500/10 text-yellow-500 hover:bg-yellow-500/20": trial.status === "recruiting",
                        "bg-blue-500/10 text-blue-500 hover:bg-blue-500/20": trial.status === "completed"
                      })}>
                        {trial.phase}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <div className="flex items-center gap-1 cursor-help">
                                <p className="text-muted-foreground">Enrollment</p>
                                <Info className="h-3 w-3 text-muted-foreground" />
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Current enrollment progress vs target</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        <p className="font-medium">
                          {trial.participant_count}/{trial.target_participant_count}
                        </p>
                      </div>
                      <div>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <div className="flex items-center gap-1 cursor-help">
                                <p className="text-muted-foreground">Safety Signals</p>
                                <Info className="h-3 w-3 text-muted-foreground" />
                              </div>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>Number of detected safety concerns</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                        <p className="font-medium">{trial.real_time_metrics.safety_signals.length}</p>
                      </div>
                    </div>
                  </div>
                )) : (
                  <div className="flex items-center justify-center h-32 text-muted-foreground">
                    No active trials available
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <Activity className="h-6 w-6 mb-4 text-primary" />
            <h3 className="font-semibold mb-2">Automated Testing</h3>
            <p className="text-sm text-muted-foreground">High-throughput screening and analysis</p>
          </Card>
          <Card className="p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <LineChart className="h-6 w-6 mb-4 text-primary" />
            <h3 className="font-semibold mb-2">Trial Analytics</h3>
            <p className="text-sm text-muted-foreground">Real-time monitoring and predictions</p>
          </Card>
          <Card className="p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <Users className="h-6 w-6 mb-4 text-primary" />
            <h3 className="font-semibold mb-2">Patient Cohorts</h3>
            <p className="text-sm text-muted-foreground">Stratified analysis and outcomes</p>
          </Card>
          <Card className="p-6 bg-card/50 hover:bg-card/80 transition-all duration-300 shadow-md hover:shadow-lg border-primary/20">
            <Boxes className="h-6 w-6 mb-4 text-primary" />
            <h3 className="font-semibold mb-2">Supply Chain</h3>
            <p className="text-sm text-muted-foreground">Demand prediction and optimization</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
