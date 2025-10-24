import { format } from "date-fns";
import { User } from "../types";

interface ProfileCardProps {
  user: User;
  onLogout: () => void;
}

export default function ProfileCard({ user, onLogout }: ProfileCardProps) {
  const joinedAt = format(new Date(user.created_at), "PPP");

  return (
    <div className="profile-card">
      <div>
        <p className="eyebrow">Profile</p>
        <h2>{user.name}</h2>
        <p className="muted">
          @{user.username}
          {user.email ? ` · ${user.email}` : null}
        </p>
      </div>
      <div className="profile-meta">
        <div>
          <span className="eyebrow">Member since</span>
          <p>{joinedAt}</p>
        </div>
        <button className="btn danger" onClick={onLogout}>
          Log out
        </button>
      </div>
    </div>
  );
}
