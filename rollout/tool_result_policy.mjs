export const name = "rollout-tool-results";
export const inject = ["toolResultPruner"];

export function apply(ctx) {
  ctx.on("agent/pre-step", ({ agent, signal }, next) => {
    if (!signal.aborted) ctx.get("toolResultPruner").pruneSession(agent.session);
    return next();
  }, { prepend: true });
}
