import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Production is mounted below an existing shared-domain reverse proxy;
  // local development leaves this empty and still runs at `/`.
  basePath: process.env.NEXT_PUBLIC_OPERATOR_BASE_PATH || undefined,
};

export default nextConfig;
