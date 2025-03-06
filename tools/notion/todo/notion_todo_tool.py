import asyncio
from langchain.tools import BaseTool
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from tools.notion.todo.notion_todo_manager import NotionTodoManager

class NotionTodoInput(BaseModel):
    """Eingabemodell für das Notion To-Do Tool."""
    action: str = Field(description="Aktion: 'get_tasks', 'get_daily_top_tasks' oder 'add_task'.")
    task_name: Optional[str] = Field(default=None, description="Name der Aufgabe (nur für 'add_task').")

class NotionTodoTool(BaseTool):
    """Verwaltet To-Do-Listen in Notion: Aufgaben abrufen, hinzufügen oder Top-Prioritäten anzeigen."""

    name: str = "notion_todo"
    description: str = "Verwalte deine Notion To-Do-Liste: Aufgaben abrufen, hinzufügen oder Top-Prioritäten anzeigen."
    args_schema: Type[BaseModel] = NotionTodoInput

    todo_manager: NotionTodoManager = Field(default_factory=NotionTodoManager, exclude=True)
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        
        self._action_map = {
            "get_tasks": self._get_tasks,
            "get_daily_top_tasks": self._get_daily_top_tasks,
            "add_task": self._add_task
        }

    def _run(self, action: str, task_name: Optional[str] = None) -> str:
        return asyncio.run(self._arun(action, task_name))

    async def _arun(self, action: str, task_name: Optional[str] = None) -> str:
        """Asynchrone Methode zur Verwaltung der Notion To-Do-Liste."""
        try:
            if action in self._action_map:
                return await self._action_map[action](task_name) 
            return f"Fehler: Unbekannte Aktion '{action}'. Erlaubt sind {list(self._action_map.keys())}."
        except Exception as e:
            return f"Fehler beim Ausführen von NotionTodoTool: {str(e)}"
        
    async def _get_tasks(self, _: Optional[str] = None) -> str:
        tasks = await self.todo_manager.get_all_todos()
        return "Keine Aufgaben gefunden." if not tasks else f"Aktuelle Aufgaben: {tasks}"

    async def _get_daily_top_tasks(self, _: Optional[str] = None) -> str:
        top_tasks = await self.todo_manager.get_daily_top_tasks()
        return "Keine Top-Prioritätsaufgaben." if not top_tasks else f"Top-Aufgaben: {top_tasks}"

    async def _add_task(self, task_name: Optional[str]) -> str:
        if not task_name:
            return "Fehler: 'task_name' ist erforderlich für 'add_task'."
        result = await self.todo_manager.add_todo(task_name)
        return f"Erfolgreich hinzugefügt: {result}"