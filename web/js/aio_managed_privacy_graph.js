// Product graph primitives shared by inactive AIO managed-privacy adapters.

export function aioManagedNodeType(node) {
  return node?.comfyClass ?? node?.type;
}

export function aioManagedGraphNodes(node) {
  const graph = node?.graph;
  if (Array.isArray(graph?._nodes)) return graph._nodes;
  if (Array.isArray(graph?.nodes)) return graph.nodes;
  return [];
}

export function aioManagedGraphLink(node, linkId) {
  const links = node?.graph?.links;
  if (links instanceof Map) return links.get(linkId);
  return links?.[linkId] ?? links?.[String(linkId)];
}

export function aioManagedOutgoingTargets(node) {
  const nodes = aioManagedGraphNodes(node);
  const targets = [];
  for (const linkId of node?.outputs?.flatMap((output) => output?.links || []) || []) {
    const link = aioManagedGraphLink(node, linkId);
    const target = nodes.find(
      (candidate) => String(candidate?.id) === String(link?.target_id),
    );
    if (target) targets.push(target);
  }
  return targets;
}
