return {
    Cite = function(el)
      if quarto.doc.is_format("latex") then
        local cites = el.citations:map(function(cite)
          return cite.id
        end)
        local citesStr = "\\cite{" .. table.concat(cites, ", ") .. "}"
        return pandoc.RawInline("latex", citesStr)
      end
    end
}