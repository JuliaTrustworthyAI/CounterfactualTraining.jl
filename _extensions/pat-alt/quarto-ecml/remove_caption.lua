-- remove_caption.lua
return {
  {
    RawBlock = function(el)
      if el.format == "tex" or el.format == "latex" then
        local content = el.text
        
        -- Remove the caption and subcaption package commands
        content = content:gsub("\\@ifpackageloaded%s*{caption}%s*{}%s*{%s*\\usepackage%s*{caption}%s*}", "")
        content = content:gsub("\\@ifpackageloaded%s*{subcaption}%s*{}%s*{%s*\\usepackage%s*{subcaption}%s*}", "")
        
        -- Only return a modified element if changes were made
        if content ~= el.text then
          el.text = content
          return el
        end
      end
    end,
    
    RawInline = function(el)
      if el.format == "tex" or el.format == "latex" then
        local content = el.text
        
        -- Remove the caption and subcaption package commands
        content = content:gsub("\\@ifpackageloaded%s*{caption}%s*{}%s*{%s*\\usepackage%s*{caption}%s*}", "")
        content = content:gsub("\\@ifpackageloaded%s*{subcaption}%s*{}%s*{%s*\\usepackage%s*{subcaption}%s*}", "")
        
        -- Only return a modified element if changes were made
        if content ~= el.text then
          el.text = content
          return el
        end
      end
    end,
    
    -- For handling the header content
    Meta = function(meta)
      -- Process header-includes if they exist
      if meta["header-includes"] then
        local header = pandoc.utils.stringify(meta["header-includes"])
        
        -- Remove the caption and subcaption package commands
        header = header:gsub("\\@ifpackageloaded%s*{caption}%s*{}%s*{%s*\\usepackage%s*{caption}%s*}", "")
        header = header:gsub("\\@ifpackageloaded%s*{subcaption}%s*{}%s*{%s*\\usepackage%s*{subcaption}%s*}", "")
        
        -- Update header-includes if changes were made
        if header ~= pandoc.utils.stringify(meta["header-includes"]) then
          meta["header-includes"] = pandoc.MetaBlocks{pandoc.RawBlock("latex", header)}
        end
      end
      
      return meta
    end
  }
}