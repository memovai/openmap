/** Central domain vocabulary for LLM normalization and test/eval heuristics.
 * LLM extraction may emit new tags; this vocabulary is the auditable schema,
 * normalization, and deterministic test surface, not the only source of truth. */

export const ALLOWED_GOALS = [
  "date",
  "romance",
  "work",
  "study",
  "family",
  "celebration",
  "business",
  "hangout",
  "solo",
  "explore",
];

export const GOAL_ROUTINE_FAMILIES = [
  { id: "business", label: "business", intents: ["business"] },
  { id: "date", label: "date", intents: ["date", "romance"] },
  { id: "family", label: "family", intents: ["family"] },
  { id: "focus", label: "focus", intents: ["work", "study"] },
  { id: "social", label: "social", intents: ["hangout", "celebration"] },
  { id: "explore", label: "explore", intents: ["solo", "explore"] },
] as const;

export const FAMILY_COMPANIONS = new Set(["parents", "kids", "family"]);
export const FAMILY_INCOMPATIBLE_GOALS = new Set(["date", "romance"]);

export const CONCEPT_LEXICON: Record<string, string[]> = {
  coffee: ["coffee", "cafe", "café", "espresso", "latte", "cappuccino", "flat white"],
  tea: ["tea", "matcha", "boba", "bubble tea"],
  ramen: ["ramen", "noodle", "noodles"],
  sushi: ["sushi", "sashimi", "omakase"],
  pizza: ["pizza", "pizzeria"],
  burger: ["burger", "burgers"],
  bar: ["bar", "pub", "cocktail", "cocktails", "beer", "brewery"],
  wine: ["wine", "wine bar", "winery"],
  brunch: ["brunch", "breakfast"],
  dessert: ["dessert", "cake", "bakery", "pastry", "ice cream", "gelato"],
  bbq: ["bbq", "barbecue", "grill"],
  vegetarian: ["vegetarian", "vegan", "plant-based"],
  open_late: ["open late", "late-night", "late night", "open now", "24/7", "after midnight"],
  walkable: ["walkable", "walk", "walking distance", "on foot"],
  low_crowd: ["uncrowded", "not crowded", "not busy", "no crowds", "no crowd", "no line", "no queue"],
  crowded: ["crowded", "busy", "packed", "crowd", "crowds", "line", "queue"],
  transit: ["transit", "station", "subway", "metro", "train", "bus"],
  parking: ["parking", "drive", "driving", "car", "valet"],
  cozy: ["cozy", "cosy", "intimate", "snug"],
  quiet: ["quiet", "calm", "peaceful", "low noise", "low-noise"],
  loud: ["loud", "noisy", "noise", "too noisy", "loud music"],
  romantic: ["romantic", "candlelit"],
  lively: ["lively", "bustling", "vibrant"],
  outdoor: ["outdoor", "patio", "terrace", "rooftop", "garden seating"],
  cheap: ["cheap", "budget", "affordable"],
  fancy: ["fancy", "upscale", "fine dining", "michelin"],
};

export const TAG_TO_CONCEPT: Record<string, string> = {
  cafe: "coffee",
  coffee_shop: "coffee",
  coffee: "coffee",
  tea: "tea",
  bar: "bar",
  pub: "bar",
  biergarten: "bar",
  bakery: "dessert",
  pastry: "dessert",
  ice_cream: "dessert",
  confectionery: "dessert",
  wine: "wine",
  ramen: "ramen",
  sushi: "sushi",
  pizza: "pizza",
  cozy: "cozy",
  quiet: "quiet",
  calm: "quiet",
  loud: "loud",
  noisy: "loud",
  romantic: "romantic",
  lively: "lively",
  outdoor: "outdoor",
  patio: "outdoor",
  terrace: "outdoor",
  vegan: "vegetarian",
  vegetarian: "vegetarian",
  plant_based: "vegetarian",
  open_late: "open_late",
  late_night: "open_late",
  walkable: "walkable",
  low_crowd: "low_crowd",
  uncrowded: "low_crowd",
  crowded: "crowded",
  busy: "crowded",
  transit: "transit",
  station: "transit",
  subway: "transit",
  metro: "transit",
  train: "transit",
  bus: "transit",
  parking: "parking",
  valet: "parking",
};

export const INTENT_LEXICON: Record<string, string[]> = {
  date: ["romantic", "intimate", "candlelit", "date", "date night", "couple"],
  romance: ["romantic", "anniversary", "candlelit"],
  work: ["work", "laptop", "wifi", "wi-fi", "remote", "co-working", "coworking"],
  study: ["study", "studying", "library"],
  family: ["family", "kid", "kids", "children", "child-friendly", "family-friendly"],
  celebration: ["birthday", "celebrate", "celebration", "party", "anniversary"],
  business: ["business", "client", "meeting", "professional"],
  hangout: ["friends", "group", "hang out", "hangout", "casual", "catch up"],
  solo: ["solo", "alone", "by myself"],
};

export const VIBE_CONCEPTS = new Set(["cozy", "quiet", "loud", "romantic", "lively", "outdoor", "cheap", "fancy"]);
export const NEGATIVE_AFFORDANCE_CONCEPTS = new Set(["loud", "crowded", "touristy", "cramped", "expensive"]);
export const LOW_CROWD_TERMS = new Set(["low_crowd", "uncrowded", "calm"]);
export const LOUD_TERMS = new Set(["loud", "noisy", "noise"]);
export const CROWDED_TERMS = new Set(["crowded", "busy", "packed"]);

export const NAME_DERIVED_CONCEPTS = new Set([
  "coffee", "tea", "ramen", "sushi", "pizza", "burger", "bar", "wine", "brunch", "dessert", "bbq", "vegetarian",
]);
export const GENERIC_PLACE_SUFFIX_TERMS = ["coffee", "cafe", "café"];

export const CONCEPT_SUMMARY_PRIORITY = [
  "quiet", "low_crowd", "transit", "walkable", "open_late", "vegetarian", "coffee", "tea", "cozy", "outdoor",
  "romantic", "lively", "cheap", "fancy", "ramen", "sushi", "pizza", "burger", "bar", "wine", "brunch",
  "dessert", "bbq", "parking", "loud", "crowded",
];
export const UTILITY_CONCEPTS = new Set(["transit", "walkable", "open_late", "vegetarian", "parking"]);

export const PLACE_ATTRIBUTE_RAW_KEYS = new Set([
  "description",
  "summary",
  "notes",
  "review",
  "reviews",
  "vibe",
  "features",
  "attributes",
  "amenities",
]);

export const MEASURE_TERMS = ["near", "walk_time", "budget", "noise", "crowd", "transit_walk"] as const;
export type MeasureTerm = typeof MEASURE_TERMS[number];
export const SCOPED_FALLBACK_MEASURE_TERMS = ["budget", "walk_time", "transit_walk"] as const;

export const RAW_NUMERIC_KEYS = {
  noise: ["noise", "noiseLevel", "noise_level", "noiseScore", "noise_score"],
  noiseDb: ["noiseDb", "noise_db", "db"],
  crowd: ["crowd", "crowdLevel", "crowd_level", "crowdScore", "crowd_score", "busyness"],
  transitWalk: ["transitWalkMin", "transit_walk_min", "stationWalkMin", "station_walk_min", "metroWalkMin", "metro_walk_min"],
  walkTime: ["walkTimeMin", "walk_time_min", "walkMin", "walk_min"],
};

export const CONSTRAINT_MATCH_TERMS = {
  openNow: ["open_late", "late night", "late-night", "open late", "24/7"],
  walkable: ["walkable", "walking distance", "on foot"],
  noise: {
    quiet: ["quiet", "calm", "peaceful", "low_noise", "low noise", "not noisy", "not loud", "not too noisy", "not too loud", "without noise", "avoid noise", "avoid noisy", "no loud music"],
    loud: ["loud", "noisy", "noise", "loud_music", "loud music"],
  },
  crowd: {
    low: ["low_crowd", "uncrowded", "not crowded", "not busy", "not too crowded", "not too busy", "without crowd", "without crowds", "avoid crowd", "avoid crowds", "no crowd", "no crowds", "no line", "no queue"],
    high: ["crowded", "busy", "packed", "crowd", "line", "queue"],
  },
  travelMode: {
    transit: ["transit", "station", "subway", "metro", "train", "bus"],
    drive: ["parking", "valet", "drive", "driving", "car access"],
  },
  dietary: ["vegetarian", "vegan", "plant-based"],
  budget: {
    low: ["cheap", "budget", "affordable"],
    high: ["fancy", "upscale", "fine dining", "michelin"],
  },
};

export const FRAME_CONSTRAINT_CONCEPTS = {
  budget: { low: "cheap", high: "fancy" },
  dietary: ["vegetarian"],
  openNow: "open_late",
  walkable: "walkable",
  noise: { quiet: "quiet", loud: "loud" },
  crowd: { low: "low_crowd", high: "crowded" },
  travelMode: { transit: "transit", drive: "parking" },
};

export interface ConstraintVocabularyInput {
  openNow?: boolean | null;
  walkable?: boolean | null;
  dietary?: string[];
  maxBudget?: "low" | "mid" | "high" | string | null;
  noise?: "quiet" | "moderate" | "loud" | string | null;
  crowd?: "low" | "moderate" | "high" | string | null;
  travelMode?: "walk" | "transit" | "drive" | string | null;
}

export function constraintConceptTerms(c: ConstraintVocabularyInput): string[] {
  return [
    ...(c.openNow ? ["open_late"] : []),
    ...(c.walkable ? ["walkable"] : []),
    ...(c.dietary ?? []),
    ...(c.maxBudget ? [c.maxBudget === "low" ? "cheap" : c.maxBudget === "high" ? "fancy" : "mid_budget"] : []),
    ...(c.noise === "quiet" ? ["quiet", "low_noise"] : c.noise === "loud" ? ["loud", "noisy"] : c.noise === "moderate" ? ["moderate_noise"] : []),
    ...(c.crowd === "low" ? ["low_crowd", "uncrowded"] : c.crowd === "high" ? ["crowded", "busy"] : c.crowd === "moderate" ? ["moderate_crowd"] : []),
    ...(c.travelMode === "walk" ? ["walkable"] : c.travelMode === "transit" ? ["transit", "station"] : c.travelMode === "drive" ? ["parking", "drive"] : []),
  ];
}

export function conceptLabel(c: string): string {
  return c === "low_crowd" ? "low crowd" : c === "open_late" ? "open late" : c === "transit" ? "near transit" : c.replace(/_/g, " ");
}
