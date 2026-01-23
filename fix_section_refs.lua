function Link(el)
	-- Handle cross-references with reference-type attribute
	if el.attributes["reference-type"] == "ref" and el.attributes["reference"] then
		local ref_id = el.attributes["reference"]

		-- Check for known prefixes and adjust accordingly
		if ref_id:match("^eq:") then
			return pandoc.Str("@eq-" .. ref_id:sub(4))
		elseif ref_id:match("^fig:") then
			return pandoc.Str("@fig-" .. ref_id:sub(5))
		elseif ref_id:match("^fig-") then
			return pandoc.Str("@fig-" .. ref_id:sub(5))
		elseif ref_id:match("^tbl:") then
			return pandoc.Str("@tbl-" .. ref_id:sub(5))
		elseif ref_id:match("^tbl-") then
			return pandoc.Str("@tbl-" .. ref_id:sub(5))
		elseif ref_id:match("^tab:") then
			return pandoc.Str("@tbl-" .. ref_id:sub(5))
		elseif ref_id:match("^tab-") then
			return pandoc.Str("@tbl-" .. ref_id:sub(5))
		elseif ref_id:match("^def:") then
			return pandoc.Str("@def-" .. ref_id:sub(5))
		else
			-- Default to section reference if no known prefix
			return pandoc.Str("@sec-" .. ref_id)
		end
	end

	-- Handle citation pattern: \protect\hyperlink{ref-KEY}{{[}number{]}}
	-- Check if this is a citation hyperlink (starts with #ref-)
	if el.target and el.target:match("^#ref%-") then
		-- Extract the citation key (remove "#ref-" prefix)
		local citation_key = el.target:sub(6)

		-- Check if the content structure resembles a citation number in brackets
		if el.content and #el.content >= 3 then
			-- Look for the pattern where there's a bracket, then a number, then a bracket
			local is_citation = false

			-- Check if first element contains opening bracket
			if el.content[1].t == "Span" and #el.content[1].content == 1 and el.content[1].content[1].text == "[" then
				-- Check if middle element is a number
				if el.content[2].t == "Str" and el.content[2].text:match("^%d+$") then
					-- Check if last element contains closing bracket
					if
						el.content[3].t == "Span"
						and #el.content[3].content == 1
						and el.content[3].content[1].text == "]"
					then
						is_citation = true
					end
				end
			end

			if is_citation then
				return pandoc.Str("@" .. citation_key)
			end
		end
	end
end
