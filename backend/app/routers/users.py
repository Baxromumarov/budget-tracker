from fastapi import APIRouter, Depends

from ..models import UserModel
from ..schemas import ProfileOut
from ..security import get_current_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=ProfileOut)
def read_profile(current_user: UserModel = Depends(get_current_user)) -> ProfileOut:
    return ProfileOut.model_validate(current_user)
