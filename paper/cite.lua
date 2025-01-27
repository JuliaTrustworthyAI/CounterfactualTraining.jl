return {
  Cite = function(el)
      -- Only process if we're in LaTeX format
      if quarto.doc.is_format("latex") then
          -- Check if this is a real citation and not a cross-reference
          local isCitation = false
          
          -- Most cross-references will have specific prefixes like 'fig:' or 'tbl:'
          for _, cite in ipairs(el.citations) do
              local id = cite.id
              -- Check if the ID doesn't start with common cross-reference prefixes
              if not (id:match("^fig-") or 
                     id:match("^tbl-") or 
                     id:match("^eq-") or 
                     id:match("^sec-")) then
                  isCitation = true
                  break
              end
          end
          
          -- Only process if it's a real citation
          if isCitation then
              local cites = el.citations:map(function(cite)
                  return cite.id
              end)
              local citesStr = "\\cite{" .. table.concat(cites, ", ") .. "}"
              return pandoc.RawInline("latex", citesStr)
          end
      end
      -- Return unchanged if not a citation or not LaTeX format
      return el
  end
}