Cite = function(el)
  -- Check if the document is being rendered to LaTeX
  if quarto.doc.is_format("latex") then
    -- Check if any citation ID does NOT contain a dash
    -- (indicating it's likely a bibliographic citation)
    local isCitation = false
    for _, cite in ipairs(el.citations) do
      if not cite.id:match("-") then
        isCitation = true
        break
      end
    end
    
    if isCitation then
      local cites = el.citations:map(function(cite)
        return cite.id
      end)
      local citesStr = "\\cite{" .. table.concat(cites, ",") .. "}"
      return pandoc.RawInline("latex", citesStr)
    end
  end
  return el
end

return {
  Cite=Cite
}