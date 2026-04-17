"use client";

import React, { useEffect, useState, useMemo, useCallback, useRef, memo } from "react";
import Link from "next/link";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  ZoomableGroup,
} from "react-simple-maps";
import { geoAlbersUsa } from "d3-geo";
import {
  getAirportCoords,
  getAllPairs,
  getNetworkSummary,
  classifyRisk,
  riskBreakdown,
  calibrateConfidence,
  fetchMetar,
  adjustedRisk,
  DFW_COORD,
  FLT_CAT_COLORS,
  type AirportCoord,
  type SlimPair,
  type NetworkSummary,
  type MetarData,
} from "@/lib/data";
// AsciiBackground removed — was causing repeating text pattern bug
import {
  Plane,
  Shield,
  AlertTriangle,
  Clock,
  ArrowDown,
  ArrowUp,
  X,
  RotateCcw,
  ChevronDown,
  Info,
  Cloud,
  Wind,
  Eye,
  Zap,
  TrendingUp,
} from "lucide-react";

const GEO_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json";
const SEASONS = ["Spring", "Summer", "Fall", "Winter"];
const MAP_W = 960;
const MAP_H = 600;

interface PopupState {
  code: string;
  x: number;
  y: number;
}

/* ---- Memoized sub-components to prevent re-renders on zoom ---- */

const MemoGeographies = memo(function MemoGeographies() {
  return (
    <Geographies geography={GEO_URL}>
      {({ geographies }) =>
        geographies.map((geo) => (
          <Geography
            key={geo.rpiKey || geo.properties?.name}
            geography={geo}
            fill="#243B5C"
            stroke="#2E4A6E"
            strokeWidth={0.5}
            style={{
              default: { outline: "none", cursor: "default" },
              hover: { outline: "none", fill: "#243B5C", cursor: "default" },
              pressed: { outline: "none", cursor: "default" },
            }}
          />
        ))
      }
    </Geographies>
  );
});

const AirportDot = memo(function AirportDot({
  ap,
  isA,
  isB,
  bothSelected,
  onDotClick,
}: {
  ap: { code: string; lon: number; lat: number };
  isA: boolean;
  isB: boolean;
  bothSelected: boolean;
  onDotClick: (code: string, e: React.MouseEvent) => void;
}) {
  const isSelected = isA || isB;
  const dimmed = bothSelected && !isSelected;
  const dotColor = isA ? "#5CB8FF" : isB ? "#FF6B7A" : "#FFFFFF";
  const dotR = isSelected ? 4.5 : 2.5;

  return (
    <Marker coordinates={[ap.lon, ap.lat]}>
      {isSelected && (
        <circle r={7} fill={dotColor} opacity={0.15} />
      )}
      <circle
        r={dotR}
        fill={dotColor}
        stroke={isSelected ? "#FFFFFF" : "none"}
        strokeWidth={isSelected ? 1.2 : 0}
        opacity={dimmed ? 0.06 : 1}
        className="airport-dot"
        onClick={(e) => {
          e.stopPropagation();
          onDotClick(ap.code, e as unknown as React.MouseEvent);
        }}
      />
      <text
        textAnchor="middle"
        y={-9}
        style={{
          fill: dotColor,
          fontSize: isSelected ? 8 : 6,
          fontWeight: isSelected ? 800 : 600,
          fontFamily: "var(--font-geist-mono)",
          opacity: dimmed ? 0.04 : isSelected ? 1 : 0.6,
        }}
        className="airport-label"
      >
        {ap.code}
      </text>
    </Marker>
  );
});

/* ---- Projection (must match ComposableMap internals) ---- */
const mapProjection = geoAlbersUsa().translate([MAP_W / 2, MAP_H / 2]);

function proj(lon: number, lat: number): [number, number] | null {
  return mapProjection([lon, lat]) as [number, number] | null;
}

/* ---- Curved path helpers (works in projected pixel space) ---- */

function buildArcPath(
  from: [number, number],
  to: [number, number]
): string {
  const midX = (from[0] + to[0]) / 2;
  const midY = (from[1] + to[1]) / 2;
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const dist = Math.sqrt(dx * dx + dy * dy);
  const arcHeight = Math.min(dist * 0.25, 80);
  // Control point: offset upward (negative Y in SVG = up)
  const cpX = midX;
  const cpY = midY - arcHeight;
  return `M ${from[0]},${from[1]} Q ${cpX},${cpY} ${to[0]},${to[1]}`;
}

/** True when coordinates lie in Alaska or Hawaii (outside CONUS) */
function isNonConus(coord: { lon: number; lat: number }): boolean {
  return coord.lat > 50 || coord.lat < 25;
}

/** Safe zoom center — falls back to full-US view for AK/HI airports */
function getZoomCenter(
  airportCoord: { lon: number; lat: number } | undefined
): { center: [number, number]; zoom: number } {
  if (!airportCoord) return { center: [-96, 38], zoom: 1 };
  if (isNonConus(airportCoord)) return { center: [-96, 38], zoom: 1 };
  return {
    center: [
      (airportCoord.lon + DFW_COORD.lon) / 2,
      (airportCoord.lat + DFW_COORD.lat) / 2,
    ],
    zoom: 2.2,
  };
}

/** Safe zoom for two airports — falls back to full-US when either is AK/HI */
function getZoomCenterPair(
  a: { lon: number; lat: number },
  b: { lon: number; lat: number }
): { center: [number, number]; zoom: number } {
  if (isNonConus(a) || isNonConus(b)) return { center: [-96, 38], zoom: 1 };
  return {
    center: [(a.lon + b.lon) / 2, (a.lat + b.lat) / 2],
    zoom: 2.2,
  };
}

export default function FlightMapPage() {
  const [coords, setCoords] = useState<AirportCoord[]>([]);
  const [allPairs, setAllPairs] = useState<SlimPair[]>([]);
  const [network, setNetwork] = useState<NetworkSummary | null>(null);
  const [airportA, setAirportA] = useState<string | null>(null);
  const [airportB, setAirportB] = useState<string | null>(null);
  const [season, setSeason] = useState("Summer");
  const [popup, setPopup] = useState<PopupState | null>(null);
  const [loading, setLoading] = useState(true);
  const [mapZoom, setMapZoom] = useState(1);
  const [mapCenter, setMapCenter] = useState<[number, number]>([-96, 38]);
  const resultsRef = useRef<HTMLDivElement>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const [pathKey, setPathKey] = useState(0); // force re-animate
  const [metars, setMetars] = useState<MetarData[]>([]);
  const [metarLoading, setMetarLoading] = useState(false);
  const [showRawMetar, setShowRawMetar] = useState(false);

  const programmaticZoomRef = useRef(false);

  /** Enable smooth CSS transition, auto-disable after animation completes */
  const triggerSmoothZoom = useCallback(() => {
    const el = mapContainerRef.current;
    if (!el) return;
    programmaticZoomRef.current = true;
    el.classList.add("smooth-zoom");
    setTimeout(() => {
      el.classList.remove("smooth-zoom");
      programmaticZoomRef.current = false;
    }, 2000);
  }, []);

  useEffect(() => {
    Promise.all([getAirportCoords(), getAllPairs(), getNetworkSummary()]).then(
      ([c, p, n]) => {
        setCoords(c);
        setAllPairs(p);
        setNetwork(n);
        setLoading(false);
      }
    );
  }, []);

  // Fetch live METAR when airports change
  useEffect(() => {
    const codes = [airportA, airportB].filter(Boolean) as string[];
    if (codes.length === 0) { setMetars([]); return; }
    setMetarLoading(true);
    const timer = setTimeout(() => {
      fetchMetar(codes).then((m) => { setMetars(m); setMetarLoading(false); });
    }, 400); // debounce
    return () => clearTimeout(timer);
  }, [airportA, airportB]);

  // Sorted airport list for dropdowns
  const airportCodes = useMemo(
    () => coords.map((c) => c.code).sort(),
    [coords]
  );

  // Coord lookup
  const coordMap = useMemo(() => {
    const m: Record<string, AirportCoord> = {};
    coords.forEach((c) => (m[c.code] = c));
    return m;
  }, [coords]);

  // Look up the selected pair
  const result = useMemo(() => {
    if (!airportA || !airportB) return null;
    return (
      allPairs.find(
        (p) =>
          ((p.a === airportA && p.b === airportB) ||
            (p.a === airportB && p.b === airportA)) &&
          p.s === season
      ) ?? null
    );
  }, [allPairs, airportA, airportB, season]);

  const riskInfo = result ? classifyRisk(result.r) : null;
  const breakdown = result ? riskBreakdown(result) : [];

  // Live-adjusted risk from METAR data
  const liveRisk = useMemo(() => {
    if (!result || metars.length === 0) return null;
    return adjustedRisk(result.r, metars);
  }, [result, metars]);
  const liveRiskInfo = liveRisk ? classifyRisk(liveRisk.adjusted) : null;

  // Find METAR for each selected airport
  const metarA = useMemo(
    () => metars.find((m) => m.iataCode === airportA) ?? null,
    [metars, airportA]
  );
  const metarB = useMemo(
    () => metars.find((m) => m.iataCode === airportB) ?? null,
    [metars, airportB]
  );

  // Zoom orchestration — managed by setInbound/setOutbound, not by a useEffect
  // (removed auto-zoom effect to give full control to the callbacks)

  // Handle clicking an airport dot
  const handleDotClick = useCallback(
    (code: string, event: React.MouseEvent) => {
      if (code === "DFW") return;
      const rect = (event.currentTarget as SVGElement)
        .closest(".map-container")
        ?.getBoundingClientRect();
      if (!rect) return;
      setPopup({
        code,
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      });
    },
    []
  );

  // Ref to hold zoom-out timer so we can cancel if outbound is picked quickly
  const zoomTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Set as inbound (Airport A)
  const setInbound = useCallback(
    (code: string) => {
      setAirportA(code);
      setPopup(null);
      setPathKey((k) => k + 1);

      // Cancel any pending zoom-out from a previous inbound pick
      if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);

      const a = coordMap[code];
      if (!a) return;

      triggerSmoothZoom();

      if (airportB && code !== airportB) {
        const b = coordMap[airportB];
        if (b) {
          const z = getZoomCenterPair(a, b);
          setMapCenter(z.center);
          setMapZoom(z.zoom);
        }
      } else {
        const z = getZoomCenter(a);
        setMapCenter(z.center);
        setMapZoom(z.zoom);
        zoomTimerRef.current = setTimeout(() => {
          triggerSmoothZoom();
          setMapZoom(1);
          setMapCenter([-96, 38]);
        }, 2000);
      }
    },
    [airportB, coordMap, triggerSmoothZoom]
  );

  // Set as outbound (Airport B)
  const setOutbound = useCallback(
    (code: string) => {
      setAirportB(code);
      setPopup(null);
      setPathKey((k) => k + 1);

      if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);

      const b = coordMap[code];
      if (!b) return;

      triggerSmoothZoom();

      if (airportA && code !== airportA) {
        const a = coordMap[airportA];
        if (a) {
          const z = getZoomCenterPair(a, b);
          setMapCenter(z.center);
          setMapZoom(z.zoom);
        }
      } else {
        const z = getZoomCenter(b);
        setMapCenter(z.center);
        setMapZoom(z.zoom);
        zoomTimerRef.current = setTimeout(() => {
          triggerSmoothZoom();
          setMapZoom(1);
          setMapCenter([-96, 38]);
        }, 2000);
      }
    },
    [airportA, coordMap, triggerSmoothZoom]
  );

  const closePopup = useCallback(() => setPopup(null), []);

  // FULL RESET
  const resetAll = useCallback(() => {
    if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current);
    triggerSmoothZoom();
    setAirportA(null);
    setAirportB(null);
    setPopup(null);
    setMapZoom(1);
    setMapCenter([-96, 38]);
    setPathKey((k) => k + 1);
  }, [triggerSmoothZoom]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => { if (zoomTimerRef.current) clearTimeout(zoomTimerRef.current); };
  }, []);

  const bothSelected = !!airportA && !!airportB && airportA !== airportB;

  // Project DFW once
  const dfwPx = useMemo(() => proj(DFW_COORD.lon, DFW_COORD.lat), []);

  // Build arc paths in PROJECTED pixel space
  const arcPaths = useMemo(() => {
    const paths: {
      id: string;
      d: string;
      color: string;
      type: "inbound" | "outbound";
    }[] = [];
    if (!dfwPx) return paths;

    if (airportA) {
      const a = coordMap[airportA];
      if (a) {
        const aPx = proj(a.lon, a.lat);
        if (aPx) {
          paths.push({
            id: `arc-inbound-${airportA}-${pathKey}`,
            d: buildArcPath(aPx, dfwPx),
            color: "#0078D2",
            type: "inbound",
          });
        }
      }
    }

    if (airportB) {
      const b = coordMap[airportB];
      if (b) {
        const bPx = proj(b.lon, b.lat);
        if (bPx) {
          paths.push({
            id: `arc-outbound-${airportB}-${pathKey}`,
            d: buildArcPath(dfwPx, bPx),
            color: "#C8102E",
            type: "outbound",
          });
        }
      }
    }

    return paths;
  }, [airportA, airportB, coordMap, pathKey, dfwPx]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#C8102E] border-t-transparent" />
          <span className="text-sm text-[#6B7B8D]">Loading flight data...</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="relative min-h-screen bg-[#F8F9FA]"
      onClick={popup ? closePopup : undefined}
    >
      <div className="relative">
        {/* --- Navy top nav --- */}
        <header className="bg-[#0A1A3A] text-white">
          <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-6">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#C8102E]">
                <Plane className="h-4 w-4 text-white -rotate-45" />
              </div>
              <div>
                <span className="text-sm font-semibold tracking-tight">
                  <span className="text-[#0078D2]">Nav</span><span className="text-[#C8102E]">Risk</span>
                </span>
                <span className="ml-1.5 text-[10px] uppercase tracking-widest text-white/50">
                  AA Analytics
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-0 border border-white/15 rounded-lg overflow-hidden">
                <Link href="/" className="page-toggle-btn active">
                  Flight Map
                </Link>
                <Link href="/model" className="page-toggle-btn">
                  Model Rundown
                </Link>
              </div>
              {network && (
                <div className="flex items-center gap-3">
                  <span className="text-[10px] uppercase tracking-wider text-white/50">
                    <span className="font-mono font-bold text-white/80">{network.nodes}</span> Airports
                  </span>
                  <span className="text-[10px] uppercase tracking-wider text-white/50">
                    <span className="font-mono font-bold text-[#FF6B7A]">{network.highEdges.toLocaleString()}</span> High-Risk
                  </span>
                </div>
              )}
            </div>
            <div className="text-[10px] uppercase tracking-widest text-white/40">
              GROW 26.2
            </div>
          </div>
        </header>

        {/* --- Main --- */}
        <div className="mx-auto max-w-[1400px] px-6 py-6">
          {/* Title row */}
          <div className="mb-4 flex items-end justify-between animate-fade-in-up">
            <div>
              <h1
                className="text-4xl font-normal tracking-tight leading-none"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                <span className="text-[#0078D2]">Nav</span><span className="text-[#C8102E]">Risk</span>
              </h1>
              <p className="mt-1 text-sm text-[#6B7B8D]">
                Explore pilot crew sequence risk across the American Airlines DFW hub network
              </p>
            </div>
          </div>


          {/* ---- AIRPORT SELECTOR PANEL (full-width, own row) ---- */}
          <div
            className="mb-4 rounded-xl border border-[#0A1A3A]/8 bg-white p-5 animate-fade-in-up"
            style={{ animationDelay: "0.05s" }}
          >
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3">
                  Select Airports — Map or Dropdown
                </h3>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
                  {/* Inbound dropdown */}
                  <div className="flex items-center gap-2 flex-1 min-w-[160px]">
                    <label className="text-[10px] font-semibold uppercase tracking-wider text-[#0078D2] whitespace-nowrap flex items-center gap-1">
                      <ArrowDown className="h-3 w-3" />
                      Inbound
                    </label>
                    <select
                      className="airport-select inbound w-full"
                      value={airportA ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v) {
                          setInbound(v);
                        } else {
                          setAirportA(null);
                        }
                      }}
                    >
                      <option value="">Choose airport…</option>
                      {airportCodes.map((code) => (
                        <option key={code} value={code} disabled={code === airportB}>
                          {code}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* DFW hub indicator */}
                  <div className="flex h-9 w-12 items-center justify-center rounded-md bg-[#C8102E] text-white text-xs font-bold font-mono shrink-0">
                    DFW
                  </div>

                  {/* Outbound dropdown */}
                  <div className="flex items-center gap-2 flex-1 min-w-[160px]">
                    <label className="text-[10px] font-semibold uppercase tracking-wider text-[#C8102E] whitespace-nowrap flex items-center gap-1">
                      <ArrowUp className="h-3 w-3" />
                      Outbound
                    </label>
                    <select
                      className="airport-select outbound w-full"
                      value={airportB ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v) {
                          setOutbound(v);
                        } else {
                          setAirportB(null);
                        }
                      }}
                    >
                      <option value="">Choose airport…</option>
                      {airportCodes.map((code) => (
                        <option key={code} value={code} disabled={code === airportA}>
                          {code}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Reset button */}
                  <button className="reset-btn" onClick={resetAll}>
                    <RotateCcw className="h-3.5 w-3.5" />
                    Reset
                  </button>
                </div>

                {/* Current selection status */}
                {!bothSelected && (
                  <div className="mt-3 rounded-lg border border-dashed border-[#0A1A3A]/10 bg-[#E8ECF0]/50 px-4 py-2.5 text-center text-xs text-[#6B7B8D]">
                    {!airportA && !airportB && (
                      <>
                        Select airports from the dropdowns above or click
                        directly on the map
                      </>
                    )}
                    {airportA && !airportB && (
                      <>
                        Inbound set:{" "}
                        <span className="font-mono font-semibold text-[#0078D2]">
                          {airportA}→DFW
                        </span>
                        . Now select an{" "}
                        <span className="font-semibold text-[#C8102E]">
                          outbound
                        </span>{" "}
                        airport.
                      </>
                    )}
                    {!airportA && airportB && (
                      <>
                        Outbound set:{" "}
                        <span className="font-mono font-semibold text-[#C8102E]">
                          DFW→{airportB}
                        </span>
                        . Now select an{" "}
                        <span className="font-semibold text-[#0078D2]">
                          inbound
                        </span>{" "}
                        airport.
                      </>
                    )}
                  </div>
                )}
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_300px] items-start">
            {/* --- Left column: Map --- */}
            <div>
              {/* Map container */}
              <div
                ref={mapContainerRef}
                className="map-container relative animate-fade-in-up"
                style={{ animationDelay: "0.05s", aspectRatio: "16/10" }}
              >
                {/* Top bar */}
                <div className="absolute top-0 left-0 right-0 z-10 flex items-center gap-3 px-4 py-2.5">
                  <span className="flex items-center gap-1.5 rounded-sm bg-[#0A1A3A]/60 backdrop-blur px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-white/80 border border-white/10">
                    <span className="h-2 w-2 rounded-full bg-[#C8102E] animate-pulse" />
                    {coords.length} Airports
                  </span>
                  <span className="rounded-sm bg-[#0A1A3A]/60 backdrop-blur px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-white/80 border border-white/10">
                    Hub: DFW
                  </span>
                  {airportA && (
                    <span className="rounded-sm bg-[#0078D2]/20 backdrop-blur px-2.5 py-1 text-[10px] font-mono font-semibold text-[#5CB8FF] border border-[#0078D2]/30">
                      A: {airportA}
                    </span>
                  )}
                  {airportB && (
                    <span className="rounded-sm bg-[#C8102E]/20 backdrop-blur px-2.5 py-1 text-[10px] font-mono font-semibold text-[#FF6B7A] border border-[#C8102E]/30">
                      B: {airportB}
                    </span>
                  )}
                  {(airportA || airportB) && (
                    <button
                      onClick={resetAll}
                      className="ml-auto flex h-6 w-6 items-center justify-center rounded bg-white/10 hover:bg-white/20 transition-colors"
                      title="Reset map"
                    >
                      <X className="h-3 w-3 text-white/60" />
                    </button>
                  )}
                </div>

                {/* SVG Map */}
                <ComposableMap
                  projection="geoAlbersUsa"
                  width={MAP_W}
                  height={MAP_H}
                  style={{ width: "100%", height: "100%" }}
                >
                  <defs>
                    {/* Single lightweight glow — reused everywhere */}
                    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
                      <feGaussianBlur stdDeviation="3" />
                    </filter>
                    {/* Radial gradient for comet head */}
                    <radialGradient id="comet-blue">
                      <stop offset="0%" stopColor="#FFFFFF" stopOpacity="1" />
                      <stop offset="40%" stopColor="#5CB8FF" stopOpacity="0.6" />
                      <stop offset="100%" stopColor="#0078D2" stopOpacity="0" />
                    </radialGradient>
                    <radialGradient id="comet-red">
                      <stop offset="0%" stopColor="#FFFFFF" stopOpacity="1" />
                      <stop offset="40%" stopColor="#FF6B7A" stopOpacity="0.6" />
                      <stop offset="100%" stopColor="#C8102E" stopOpacity="0" />
                    </radialGradient>
                  </defs>

                  <ZoomableGroup
                    center={mapCenter}
                    zoom={mapZoom}
                    minZoom={1}
                    maxZoom={5}
                    onMoveEnd={({ coordinates, zoom }) => {
                      if (programmaticZoomRef.current) return;
                      setMapCenter(coordinates as [number, number]);
                      setMapZoom(zoom);
                    }}
                  >
                    {/* State outlines — memoized for performance */}
                    <MemoGeographies />

                    {/* ---- ANIMATED CURVED FLIGHT PATHS ---- */}
                    {arcPaths.map((arc) => {
                      const isBlue = arc.type === "inbound";
                      const bright = isBlue ? "#5CB8FF" : "#FF6B7A";
                      const comet = isBlue ? "url(#comet-blue)" : "url(#comet-red)";

                      return (
                        <g key={arc.id} className="beam-group">
                          {/* Glow layer (wide, soft) */}
                          <path
                            d={arc.d}
                            fill="none"
                            stroke={arc.color}
                            strokeWidth={8}
                            strokeLinecap="round"
                            pathLength={1}
                            className="beam-glow"
                          />
                          {/* Core beam (bright, shoots in) */}
                          <path
                            d={arc.d}
                            fill="none"
                            stroke={bright}
                            strokeWidth={2.5}
                            strokeLinecap="round"
                            pathLength={1}
                            className="beam-core"
                          />
                          {/* White center highlight */}
                          <path
                            d={arc.d}
                            fill="none"
                            stroke="#FFFFFF"
                            strokeWidth={0.8}
                            strokeLinecap="round"
                            pathLength={1}
                            className="beam-white"
                          />
                          {/* Glowing plane traveling along the arc */}
                          <g className="traveling-plane">
                            {/* Halo glow behind the plane */}
                            <circle r={8} fill={comet} opacity={0.4} filter="url(#glow)">
                              <animateMotion dur="3s" repeatCount="indefinite" path={arc.d} rotate="auto" />
                            </circle>
                            {/* Top-down airplane silhouette — nose points right (+x) */}
                            <g fill="#FFFFFF" opacity={0.92}>
                              <animateMotion dur="3s" repeatCount="indefinite" path={arc.d} rotate="auto" />
                              <g transform="scale(0.55)">
                                {/* Fuselage */}
                                <ellipse cx="0" cy="0" rx="10" ry="2.2" />
                                {/* Left wing */}
                                <polygon points="2,-2 -2,-2 -5,-10 -3,-10" />
                                {/* Right wing */}
                                <polygon points="2,2 -2,2 -5,10 -3,10" />
                                {/* Left engine */}
                                <ellipse cx="-3" cy="-7" rx="2" ry="0.9" />
                                {/* Right engine */}
                                <ellipse cx="-3" cy="7" rx="2" ry="0.9" />
                                {/* Left horizontal stabilizer */}
                                <polygon points="-7,-2 -9,-2 -10,-5 -8,-5" />
                                {/* Right horizontal stabilizer */}
                                <polygon points="-7,2 -9,2 -10,5 -8,5" />
                                {/* Nose highlight */}
                                <ellipse cx="8" cy="0" rx="2.5" ry="1.4" fill="#FFFFFF" opacity="0.5" />
                              </g>
                            </g>
                          </g>
                        </g>
                      );
                    })}

                    {/* DFW hub */}
                    <Marker
                      coordinates={[DFW_COORD.lon, DFW_COORD.lat]}
                    >
                      <circle r={10} fill="#FFFFFF" opacity={0.06} />
                      <circle r={5} fill="#FFFFFF" stroke="#C8102E" strokeWidth={2} className="dfw-dot" />
                      <text
                        textAnchor="middle"
                        y={-11}
                        style={{
                          fill: "#FFFFFF",
                          fontSize: 9,
                          fontWeight: 800,
                          fontFamily: "var(--font-geist-mono)",
                        }}
                        className="airport-label-hub"
                      >
                        DFW
                      </text>
                    </Marker>

                    {/* Airport dots — memoized per-dot to avoid full-list re-renders */}
                    {coords.map((ap) => (
                      <AirportDot
                        key={`dot-${ap.code}`}
                        ap={ap}
                        isA={ap.code === airportA}
                        isB={ap.code === airportB}
                        bothSelected={bothSelected}
                        onDotClick={handleDotClick}
                      />
                    ))}
                  </ZoomableGroup>
                </ComposableMap>

                {/* Click popup */}
                {popup && (
                  <div
                    className="airport-popup absolute animate-scale-in"
                    style={{
                      left: popup.x,
                      top: popup.y - 10,
                      transform: "translate(-50%, -100%)",
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="px-3 py-1.5 border-b border-[#0A1A3A]/6 bg-[#E8ECF0]/50">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-[#0A1A3A]">
                        {popup.code}
                      </span>
                    </div>
                    <button
                      className="airport-popup-btn inbound"
                      onClick={() => setInbound(popup.code)}
                    >
                      <ArrowDown className="h-3.5 w-3.5" />
                      <span>Set as Inbound (A→DFW)</span>
                    </button>
                    <button
                      className="airport-popup-btn outbound"
                      onClick={() => setOutbound(popup.code)}
                    >
                      <ArrowUp className="h-3.5 w-3.5" />
                      <span>Set as Outbound (DFW→B)</span>
                    </button>
                  </div>
                )}

                {/* Bottom bar */}
                <div className="map-bottom-bar absolute bottom-0 left-0 right-0">
                  <div className="flex items-center gap-6">
                    <div>
                      <span className="text-[10px] uppercase tracking-wider text-white/40">
                        Historical
                      </span>
                      <span className="ml-2 font-mono text-base font-bold text-white">
                        {result ? result.r.toFixed(2) : "—"}
                      </span>
                    </div>
                    {liveRisk && liveRisk.multiplier > 1 && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-[#FF6B7A]">
                          Live-Adj
                        </span>
                        <span className="ml-2 font-mono text-base font-bold text-[#FF6B7A]">
                          {liveRisk.adjusted.toFixed(2)}
                        </span>
                      </div>
                    )}
                    <div>
                      <span className="text-[10px] uppercase tracking-wider text-white/40">
                        Confidence
                      </span>
                      <span className="ml-2 font-mono text-base font-bold text-white">
                        {result ? `${calibrateConfidence(result.c).toFixed(0)}%` : "—"}
                      </span>
                    </div>
                  </div>
                  {bothSelected && (liveRiskInfo ?? riskInfo) && (
                    <span
                      className="rounded-sm px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider"
                      style={{
                        color: (liveRiskInfo ?? riskInfo)!.color,
                        backgroundColor: (liveRiskInfo ?? riskInfo)!.bgColor,
                      }}
                    >
                      {(liveRiskInfo ?? riskInfo)!.label}
                    </span>
                  )}
                </div>

                {/* Legend */}
                <div className="absolute bottom-12 right-3 flex flex-col gap-1.5 text-[9px] text-white/50">
                  <span className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-[#5CB8FF]" />{" "}
                    Inbound (A→DFW)
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-[#FF6B7A]" />{" "}
                    Outbound (DFW→B)
                  </span>
                </div>
              </div>

            </div>

            {/* --- Right sidebar: Season + Cards --- */}
            <div
              className="space-y-4 animate-fade-in-up"
              style={{ animationDelay: "0.1s" }}
            >
              {/* Season selector */}
              <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-4">
                <span className="text-[10px] uppercase tracking-wider text-[#6B7B8D] font-semibold">
                  Season
                </span>
                <div className="mt-2 flex gap-0 border border-[#0A1A3A]/10 rounded-md overflow-hidden">
                  {SEASONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => setSeason(s)}
                      className={`flex-1 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                        season === s
                          ? "bg-[#0A1A3A] text-white"
                          : "text-[#6B7B8D] hover:bg-[#E8ECF0]"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Prompt / pair not found */}
              {!bothSelected && (
                <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-8 text-center">
                  <Plane className="h-10 w-10 mx-auto mb-3 text-[#6B7B8D]/20 -rotate-45" />
                  <h3 className="text-sm font-semibold mb-1">
                    Select a Crew Sequence
                  </h3>
                  <p className="text-xs text-[#6B7B8D] leading-relaxed max-w-[240px] mx-auto">
                    Click any airport on the map or use the dropdowns below to
                    choose the{" "}
                    <span className="text-[#0078D2] font-semibold">
                      inbound
                    </span>{" "}
                    and{" "}
                    <span className="text-[#C8102E] font-semibold">
                      outbound
                    </span>{" "}
                    legs of the sequence through DFW.
                  </p>
                </div>
              )}
              {bothSelected && !result && (
                <div className="rounded-xl border border-[#D4880F]/30 bg-[#D4880F]/5 p-5 text-center">
                  <p className="text-sm font-semibold text-[#D4880F]">
                    Pair Not Found
                  </p>
                  <p className="text-xs text-[#6B7B8D] mt-1">
                    {airportA}→DFW→{airportB} was not scored for {season}. Try a
                    different combination.
                  </p>
                </div>
              )}
              {bothSelected && result && riskInfo && (
                <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D]">
                      Sequence Risk
                    </h3>
                    <span
                      className="rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                      style={{
                        color: riskInfo.color,
                        backgroundColor: riskInfo.bgColor,
                      }}
                    >
                      {riskInfo.label}
                    </span>
                  </div>
                  <div className="text-center mb-4">
                    <p
                      className="text-4xl font-bold font-mono"
                      style={{ color: riskInfo.color }}
                    >
                      {(result.r * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-[#6B7B8D] mt-1 font-mono">
                      {airportA} → DFW → {airportB} · {season}
                    </p>
                  </div>
                  <div className="space-y-2.5 border-t border-[#0A1A3A]/6 pt-3">
                    <MetricRow
                      icon={<Shield className="h-3.5 w-3.5 text-[#0078D2]" />}
                      label="Confidence"
                      value={`${calibrateConfidence(result.c).toFixed(0)}%`}
                    />
                    <MetricRow
                      icon={<Clock className="h-3.5 w-3.5 text-[#D4880F]" />}
                      label="Duty Violation"
                      value={`${(result.d * 100).toFixed(1)}%`}
                    />
                    <MetricRow
                      icon={<AlertTriangle className="h-3.5 w-3.5 text-[#534AB7]" />}
                      label="Turnaround Risk"
                      value={result.mt.toFixed(4)}
                    />
                  </div>
                </div>
              )}
              {/* ---- RISK SEVERITY PIE CHART (below sequence risk) ---- */}
              {bothSelected && result && riskInfo && (
                <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3">
                    Risk Severity Matrix
                  </h3>
                  <RiskPieChart breakdown={breakdown} />
                </div>
              )}
            </div>
          </div>

          {/* ---- DETAILED RESULTS (below the map row) ---- */}
          {bothSelected && result && riskInfo && (
            <div ref={resultsRef} className="mt-6 animate-fade-in-up">
              {/* ---- VERDICT BANNER (top of results) ---- */}
              <div className="mb-6 rounded-xl border-2 bg-white p-5" style={{ borderColor: riskInfo.color }}>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-1">
                      Why This Sequence Is{" "}
                      <span style={{ color: riskInfo.color }}>
                        {result.r >= 0.7 ? "Flagged" : result.r >= 0.4 ? "Cautioned" : "Accepted"}
                      </span>
                    </h3>
                    <p className="text-sm leading-relaxed text-[#3A4A5A]">
                      The XGBoost model scored{" "}
                      <span className="font-mono font-semibold">{airportA}→DFW→{airportB}</span>{" "}
                      at{" "}
                      <span className="font-mono font-semibold" style={{ color: riskInfo.color }}>
                        {(result.r * 100).toFixed(1)}%
                      </span>{" "}
                      risk. Primary driver:{" "}
                      <span className="font-semibold">
                        {breakdown.reduce((a, b) => (a.pct > b.pct ? a : b)).name.toLowerCase()}
                      </span>{" "}
                      ({breakdown.reduce((a, b) => (a.pct > b.pct ? a : b)).pct.toFixed(0)}% contribution).
                    </p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-3xl font-bold font-mono" style={{ color: riskInfo.color }}>
                      {(result.r * 100).toFixed(1)}%
                    </span>
                    <span
                      className="rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider"
                      style={{ color: riskInfo.color, backgroundColor: riskInfo.bgColor }}
                    >
                      {riskInfo.label}
                    </span>
                  </div>
                </div>
                <div
                  className="mt-3 rounded-lg p-3 text-xs leading-relaxed"
                  style={{ backgroundColor: riskInfo.bgColor, color: riskInfo.color }}
                >
                  {result.r >= 0.7
                    ? `This pair should NOT be included in pilot crew sequences. The combined duty burden and propagation risk make it dangerous for ${season.toLowerCase()} operations.`
                    : result.r >= 0.4
                    ? `This pair should be monitored closely. Consider adding turnaround buffer time during ${season.toLowerCase()} scheduling.`
                    : `This pair is generally acceptable for crew scheduling. Historical data shows manageable risk levels during ${season.toLowerCase()}.`}
                </div>
              </div>

              <div className="grid gap-6" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
                {/* ---- LIVE WEATHER CONDITIONS ---- */}
                {(metarA || metarB || metarLoading) && (
                  <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-5 h-full">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3 flex items-center gap-1.5">
                      <Cloud className="h-3.5 w-3.5" />
                      Live Weather — AWC METAR
                    </h3>
                    {metarLoading ? (
                      <div className="flex items-center gap-2 text-xs text-[#6B7B8D]">
                        <div className="h-3 w-3 animate-spin rounded-full border border-[#6B7B8D] border-t-transparent" />
                        Fetching live conditions...
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {[metarA, metarB].map((m) => {
                          if (!m) return null;
                          const isA = m.iataCode === airportA;
                          const catColors = FLT_CAT_COLORS[m.fltCat] ?? FLT_CAT_COLORS.VFR;
                          return (
                            <div key={m.icaoId} className="rounded-lg border border-[#0A1A3A]/6 p-3">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-semibold font-mono" style={{ color: isA ? "#0078D2" : "#C8102E" }}>
                                  {m.iataCode} {isA ? "(Inbound)" : "(Outbound)"}
                                </span>
                                <span
                                  className="rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                                  style={{ color: catColors.text, backgroundColor: catColors.bg }}
                                >
                                  {m.fltCat}
                                </span>
                              </div>
                              <div className="grid grid-cols-3 gap-2 text-[11px]">
                                <div className="flex items-center gap-1 text-[#6B7B8D]">
                                  <Wind className="h-3 w-3" />
                                  <span>{m.wspd ?? 0}kt{m.wgst ? ` G${m.wgst}` : ""}</span>
                                </div>
                                <div className="flex items-center gap-1 text-[#6B7B8D]">
                                  <Eye className="h-3 w-3" />
                                  <span>{m.visib ?? "—"}sm</span>
                                </div>
                                <div className="flex items-center gap-1 text-[#6B7B8D]">
                                  <Cloud className="h-3 w-3" />
                                  <span>{m.base ? `${m.base}ft` : "CLR"}</span>
                                </div>
                              </div>
                              {m.wxString && (
                                <div className="mt-1.5 flex items-center gap-1">
                                  <Zap className="h-3 w-3 text-[#D4880F]" />
                                  <span className="text-[11px] font-semibold text-[#D4880F]">{m.wxString}</span>
                                </div>
                              )}
                              <button
                                type="button"
                                className="mt-2 flex items-center gap-1 text-[10px] text-[#0078D2] hover:text-[#005FA3] font-medium transition-colors"
                                onClick={() => setShowRawMetar((v) => !v)}
                              >
                                <ChevronDown className={`h-3 w-3 transition-transform ${showRawMetar ? "rotate-180" : ""}`} />
                                {showRawMetar ? "Hide Raw Data" : "View Raw Data"}
                              </button>
                              {showRawMetar && (
                                <p className="mt-1 font-mono text-[9px] text-[#6B7B8D]/70 leading-relaxed break-all">
                                  {m.rawOb}
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {/* ---- LIVE-ADJUSTED RISK ---- */}
                {liveRisk && liveRisk.multiplier > 1 && liveRiskInfo && (
                  <div
                    className="rounded-xl border-2 p-5 h-full"
                    style={{
                      borderColor: liveRiskInfo.color,
                      backgroundColor: liveRiskInfo.bgColor,
                    }}
                  >
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3 flex items-center gap-1.5">
                      <TrendingUp className="h-3.5 w-3.5" />
                      Live-Adjusted Risk
                    </h3>
                    <div className="flex items-baseline gap-3 mb-2">
                      <p
                        className="text-3xl font-bold font-mono"
                        style={{ color: liveRiskInfo.color }}
                      >
                        {(liveRisk.adjusted * 100).toFixed(1)}%
                      </p>
                      <span className="text-xs font-mono text-[#6B7B8D] line-through">
                        {(result.r * 100).toFixed(1)}%
                      </span>
                      <span
                        className="text-xs font-bold font-mono"
                        style={{ color: liveRiskInfo.color }}
                      >
                        +{((liveRisk.adjusted - result.r) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="space-y-1">
                      {liveRisk.reasons.map((r, i) => (
                        <p key={i} className="text-[11px] text-[#3A4A5A] flex items-center gap-1.5">
                          <AlertTriangle className="h-3 w-3 shrink-0" style={{ color: liveRiskInfo.color }} />
                          {r}
                        </p>
                      ))}
                    </div>
                    <p className="mt-2 text-[10px] text-[#6B7B8D]">
                      Weather multiplier: {liveRisk.multiplier}x — sourced from aviationweather.gov METAR
                    </p>
                  </div>
                )}



                {/* Severity Matrix — Pie Chart */}
                <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3">
                    Risk Severity Matrix
                  </h3>
                  <div className="flex items-center gap-6">
                    <svg viewBox="0 0 120 120" className="h-32 w-32 shrink-0">
                      {(() => {
                        const cx = 60, cy = 60, r = 54;
                        let cumulative = 0;
                        return breakdown.map((b) => {
                          const startAngle = cumulative * 3.6 * (Math.PI / 180);
                          cumulative += b.pct;
                          const endAngle = cumulative * 3.6 * (Math.PI / 180);
                          const largeArc = b.pct > 50 ? 1 : 0;
                          const x1 = cx + r * Math.sin(startAngle);
                          const y1 = cy - r * Math.cos(startAngle);
                          const x2 = cx + r * Math.sin(endAngle);
                          const y2 = cy - r * Math.cos(endAngle);
                          if (b.pct <= 0) return null;
                          if (b.pct >= 100) {
                            return (
                              <circle key={b.name} cx={cx} cy={cy} r={r} fill={b.color} />
                            );
                          }
                          return (
                            <path
                              key={b.name}
                              d={`M ${cx},${cy} L ${x1},${y1} A ${r},${r} 0 ${largeArc} 1 ${x2},${y2} Z`}
                              fill={b.color}
                            />
                          );
                        });
                      })()}
                      <circle cx={60} cy={60} r={28} fill="white" />
                    </svg>
                    <div className="space-y-2.5 flex-1">
                      {breakdown.map((b) => (
                        <div key={b.name} className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-sm shrink-0" style={{ backgroundColor: b.color }} />
                          <span className="text-xs flex-1">{b.name}</span>
                          <span className="text-xs font-mono font-semibold">{b.pct.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>


                {/* ---- DELAY PROPAGATION EXPLANATION ---- */}
                <div className="rounded-xl border border-[#0A1A3A]/8 bg-white p-5 h-full">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#6B7B8D] mb-3 flex items-center gap-1.5">
                    <Info className="h-3.5 w-3.5" />
                    Delay Propagation Logic
                  </h3>
                  <div className="space-y-3 text-xs leading-relaxed text-[#3A4A5A]">
                    <div className="flex items-center gap-2 rounded-lg bg-[#E8ECF0] px-3 py-2 font-mono text-[11px]">
                      <span className="text-[#0078D2] font-bold">{airportA}</span>
                      <span className="text-[#6B7B8D]">→ late →</span>
                      <span className="font-bold text-[#C8102E]">DFW</span>
                      <span className="text-[#6B7B8D]">→ short turn →</span>
                      <span className="text-[#C8102E] font-bold">{airportB}</span>
                      <span className="text-[#6B7B8D]">= cascaded delay</span>
                    </div>
                    <ul className="space-y-1.5 list-disc list-inside text-[#3A4A5A]">
                      <li><span className="font-semibold text-[#0A1A3A]">Shorter Turnarounds:</span> Late inbound flights reduce crew turnaround time.</li>
                      <li><span className="font-semibold text-[#0A1A3A]">Scoring:</span> The <span className="font-mono font-semibold">{result.cp.toFixed(1)}</span> combined propagation score measures historical cascading tendencies between these specific airports.</li>
                    </ul>
                  </div>
                </div>

              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function RiskPieChart({
  breakdown,
}: {
  breakdown: { name: string; pct: number; color: string }[];
}) {
  const [activeSlice, setActiveSlice] = useState<string | null>(null);

  const size = 160;
  const cx = size / 2;
  const cy = size / 2;
  const r = 58;
  const labelR = r + 14;

  // Build SVG arc segments
  let cumulative = 0;
  const slices = breakdown.map((b) => {
    const startAngle = (cumulative / 100) * 360;
    cumulative += b.pct;
    const endAngle = (cumulative / 100) * 360;
    const midAngle = (startAngle + endAngle) / 2;
    return { ...b, startAngle, endAngle, midAngle };
  });

  function polarToCartesian(angle: number, radius: number) {
    const rad = ((angle - 90) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  const handleClick = (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setActiveSlice((prev) => (prev === name ? null : name));
  };

  return (
    <div className="flex flex-col items-center gap-3" onClick={() => setActiveSlice(null)}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {slices.map((s) => {
          if (s.pct <= 0) return null;
          const isActive = activeSlice === s.name;
          const isDimmed = activeSlice !== null && !isActive;
          const fillColor = isDimmed ? "#D1D5DB" : s.color;
          const scale = isActive ? 1.06 : 1;

          // Full circle edge case
          if (s.pct >= 100) {
            return (
              <circle
                key={s.name}
                cx={cx}
                cy={cy}
                r={r}
                fill={fillColor}
                style={{ cursor: "pointer", transition: "fill 0.2s ease" }}
                onClick={(e) => handleClick(s.name, e)}
              />
            );
          }
          const start = polarToCartesian(s.startAngle, r);
          const end = polarToCartesian(s.endAngle, r);
          const largeArc = s.endAngle - s.startAngle > 180 ? 1 : 0;
          const d = `M ${cx} ${cy} L ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y} Z`;

          // Label position at midpoint of the arc
          const labelPos = polarToCartesian(s.midAngle, labelR);

          return (
            <g key={s.name}>
              <path
                d={d}
                fill={fillColor}
                stroke="#FFFFFF"
                strokeWidth={1.5}
                style={{
                  cursor: "pointer",
                  transform: isActive ? `scale(${scale})` : "none",
                  transformOrigin: `${cx}px ${cy}px`,
                  transition: "fill 0.2s ease, transform 0.2s ease",
                }}
                onClick={(e) => handleClick(s.name, e)}
              />
              {isActive && (
                <text
                  x={labelPos.x}
                  y={labelPos.y}
                  textAnchor="middle"
                  dominantBaseline="central"
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    fontFamily: "var(--font-geist-mono)",
                    fill: s.color,
                    pointerEvents: "none",
                  }}
                >
                  {s.pct.toFixed(0)}%
                </text>
              )}
            </g>
          );
        })}
        <circle cx={cx} cy={cy} r={28} fill="#FFFFFF" />
        {activeSlice && (
          <text
            x={cx}
            y={cy}
            textAnchor="middle"
            dominantBaseline="central"
            style={{
              fontSize: 8,
              fontWeight: 600,
              fill: "#6B7B8D",
              pointerEvents: "none",
            }}
          >
            {activeSlice.length > 10 ? activeSlice.slice(0, 9) + "…" : activeSlice}
          </text>
        )}
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 w-full">
        {breakdown.map((b) => {
          const isDimmed = activeSlice !== null && activeSlice !== b.name;
          return (
            <div
              key={b.name}
              className="flex items-center gap-1.5 cursor-pointer"
              style={{ opacity: isDimmed ? 0.35 : 1, transition: "opacity 0.2s ease" }}
              onClick={(e) => handleClick(b.name, e)}
            >
              <span
                className="h-2.5 w-2.5 rounded-full shrink-0"
                style={{ backgroundColor: isDimmed ? "#D1D5DB" : b.color }}
              />
              <span className="text-[10px] text-[#3A4A5A] truncate">{b.name}</span>
              <span className="text-[10px] font-mono font-semibold ml-auto">{b.pct.toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-1.5 text-xs text-[#6B7B8D]">
        {icon}
        <span>{label}</span>
      </div>
      <span className="font-mono text-sm font-semibold">{value}</span>
    </div>
  );
}
