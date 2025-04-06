from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
import json
import os
from datetime import datetime


# Модели данных
class ExerciseInfo(BaseModel):
    название: str
    информация_о_выполнении: str = Field(..., alias="информация о выполнении")


class MealInfo(BaseModel):
    прием: str
    блюдо: str
    калории_и_БЖУ: str = Field(..., alias="калории и БЖУ")


class DayWorkout(BaseModel):
    тип_тренировки: str = Field(..., alias="тип тренировки")
    список_упражнений: List[ExerciseInfo] = Field(..., alias="список упражнений")


class DayNutrition(BaseModel):
    суточная_калорийность_БЖУ: str = Field(..., alias="суточная калорийность, БЖУ")
    приемы_пищи: List[MealInfo] = Field(..., alias="приемы пищи")


class DaySchedule(BaseModel):
    тренировки: DayWorkout
    питание: DayNutrition


class FitnessPlan(BaseModel):
    user_info: Dict[str, Any]
    schedule: Dict[str, DaySchedule]  # Ключи - дни недели

    @classmethod
    def from_raw_data(cls, data: Dict[str, Any]):
        # Преобразуем исходные данные в нужную структуру
        user_info = data["user_info"]
        schedule = {}

        # Дни недели в правильном порядке
        days_order = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]

        for day in days_order:
            if day in data:
                day_data = data[day]
                schedule[day] = {
                    "тренировки": day_data["тренировки"],
                    "питание": day_data["питание"]
                }

        return cls(user_info=user_info, schedule=schedule)

class FitnessDB:
    def __init__(self, persist_directory: str = "./fitness_db3"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2"
        )
        self.vector_store = Chroma(
            collection_name="fitness_plans",
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def _prepare_embedding_text(self, plan: FitnessPlan) -> str:
        """Только параметры пользователя для эмбеддингов"""
        return "\n".join([
            f"Пользователь: {plan.user_info['пол']}, {plan.user_info['возраст']} лет",
            f"Цель: {plan.user_info['цель']}",
            f"Параметры: рост {plan.user_info['рост']} см, вес {plan.user_info['вес']} кг"
        ])

    def add_plan(self, plan_data: Dict[str, Any]) -> str:
        """Добавление плана с детальным расписанием в метаданные"""
        plan_id = str(uuid.uuid4())
        plan = FitnessPlan.from_raw_data(plan_data)

        # Формируем структуру для метаданных
        days_schedule = {}
        for day, schedule in plan.schedule.items():
            # Информация о тренировках
            workout_info = {
                "тип_тренировки": schedule.тренировки.тип_тренировки,
                "упражнения": [
                    {
                        "название": ex.название,
                        "информация_о_выполнении": ex.информация_о_выполнении
                    } for ex in schedule.тренировки.список_упражнений
                ]
            }

            # Информация о питании
            nutrition_info = {
                "суточная_калорийность": schedule.питание.суточная_калорийность_БЖУ,
                "приемы_пищи": [
                    {
                        "прием": meal.прием,
                        "блюдо": meal.блюдо,
                        "калории_и_БЖУ": meal.калории_и_БЖУ
                    } for meal in schedule.питание.приемы_пищи
                ]
            }

            days_schedule[day] = {
                "тренировки": workout_info,
                "питание": nutrition_info
            }

        document = self._prepare_embedding_text(plan)
        metadata = {
            "id": plan_id,
            "расписание": json.dumps(days_schedule, ensure_ascii=False),  # Сериализуем в JSON
            **{k: v for k, v in plan.user_info.items() if k in ["пол", "возраст", "вес", "рост", "цель"]}
        }

        self.vector_store.add_texts(
            texts=[document],
            metadatas=[metadata],
            ids=[plan_id]
        )
        return plan_id

    def search_plans(
            self,
            user_params: Dict[str, Any],
            filter_days: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Поиск планов с детальным расписанием"""
        search_query = "\n".join([
            f"Пользователь: {user_params.get('пол', '')}, {user_params.get('возраст', '')} лет",
            f"Цель: {user_params.get('цель', '')}",
            f"Параметры: рост {user_params.get('рост', '')} см, вес {user_params.get('вес', '')} кг"
        ])

        results = self.vector_store.similarity_search(
            search_query,
            k=1
        )

        output = []
        for doc in results:
            # Десериализуем расписание из JSON
            schedule = json.loads(doc.metadata["расписание"])
            filtered_schedule = {}

            # Фильтрация по дням если нужно
            for day in schedule:
                if not filter_days or day in filter_days:
                    filtered_schedule[day] = schedule[day]

            # Добавляем отфильтрованное расписание напрямую
            output.append(filtered_schedule)

        return output

    def _check_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Проверка соответствия метаданных фильтрам"""
        if not filters:
            return True

        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def get_plan_details(self, plan_id: str) -> Dict[str, Any]:
        """Получение полной информации о плане"""
        doc = self.vector_store.get(ids=[plan_id])
        if not doc:
            return None

        return {
            "metadata": doc.metadata,
            "content": doc.page_content
        }

    def delete_plan(self, plan_id: str) -> bool:
        """Удаление плана"""
        try:
            self.vector_store.delete(ids=[plan_id])
            return True
        except Exception as e:
            print(f"Ошибка удаления: {e}")
            return False

    def persist(self):
        """Сохранение данных на диск"""
        self.vector_store.persist()