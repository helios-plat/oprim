import pytest
from datetime import date
from oprim.parse_obsidian_tasks import parse_obsidian_tasks, ObsidianTask
from oprim._exceptions import OprimError

def test_parse_obsidian_tasks_basic():
    content = "- [ ] Buy milk"
    tasks = parse_obsidian_tasks(content=content)
    assert len(tasks) == 1
    assert tasks[0].text == "Buy milk"
    assert tasks[0].completed is False

def test_parse_obsidian_tasks_completed():
    content = "- [x] Done task\n- [X] Done too"
    tasks = parse_obsidian_tasks(content=content)
    assert len(tasks) == 2
    assert all(t.completed for t in tasks)

def test_parse_obsidian_tasks_dates():
    content = "- [ ] Task 1 📅 2024-05-20 ⏳ 2024-05-19"
    tasks = parse_obsidian_tasks(content=content)
    assert tasks[0].due_date == date(2024, 5, 20)
    assert tasks[0].scheduled_date == date(2024, 5, 19)

def test_parse_obsidian_tasks_invalid_dates():
    content = "- [ ] Task 1 📅 2024-13-40"
    tasks = parse_obsidian_tasks(content=content)
    assert tasks[0].due_date is None

def test_parse_obsidian_tasks_tags():
    content = "- [ ] Multi tags #tag1 #work/urgent"
    tasks = parse_obsidian_tasks(content=content)
    assert tasks[0].tags == ["tag1", "work/urgent"]
    assert tasks[0].text == "Multi tags"

def test_parse_obsidian_tasks_none_content():
    with pytest.raises(OprimError):
        parse_obsidian_tasks(content=None)

def test_parse_obsidian_tasks_empty_text():
    content = "- [ ]   \n- [x]"
    tasks = parse_obsidian_tasks(content=content)
    assert len(tasks) == 0

def test_parse_obsidian_tasks_multiple_lines():
    content = "Some text\n- [ ] Task 1\nMore text\n- [x] Task 2"
    tasks = parse_obsidian_tasks(content=content)
    assert len(tasks) == 2
    assert tasks[0].line_number == 2
    assert tasks[1].line_number == 4

def test_parse_obsidian_tasks_example():
    content = "- [ ] Buy milk 📅 2024-05-20 #errand"
    tasks = parse_obsidian_tasks(content=content)
    assert tasks[0].text == "Buy milk"
    assert tasks[0].completed is False
    assert str(tasks[0].due_date) == "2024-05-20"
    assert tasks[0].tags == ["errand"]

def test_parse_obsidian_tasks_due_prefix():
    content = "- [ ] Task due: 2024-05-20 scheduled: 2024-05-19"
    tasks = parse_obsidian_tasks(content=content)
    assert tasks[0].due_date == date(2024, 5, 20)
    assert tasks[0].scheduled_date == date(2024, 5, 19)
