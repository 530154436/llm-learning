#!/usr/bin/env python3
# -*- coding:utf-8 -*--
# indent mapping

heading_marks = ["## ", "### ", "#### ", "##", "##", "###"]
mark2level = dict()
for hm in heading_marks:
    marks = sorted(set(map(lambda x: (x.strip(), x.count("#")), heading_marks)), key=lambda x: x[1])
    for level, (mark, mark_count) in enumerate(marks, start=1):
        mark2level[mark] = level
print(mark2level)
