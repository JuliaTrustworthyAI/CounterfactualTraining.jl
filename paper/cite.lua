function Cite(citations)
    local latex_cites = {}
    for _, citation in ipairs(citations) do
      table.insert(latex_cites, '\\cite{' .. citation.id .. '}')
    end
    -- Return the constructed LaTeX citations as a RawInline element
    return pandoc.RawInline('latex', table.concat(latex_cites, ', '))
end
  

  