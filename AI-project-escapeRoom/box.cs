using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
public class Box : GameObject
{
    public bool IsPickedUp { get; private set; }

    public Box(Vector2 position, Vector2 size) : base(position, size) { }

    public void PickUp()
    {
        IsPickedUp = true;
        gravity = 0;
    }

    public void Drop(Vector2 newPosition)
    {
        IsPickedUp = false;
        Position = newPosition;
        gravity = 9.8f;
        IsGrounded = false;
    }
    public new void Update(GameTime gameTime)
    {
        if (!IsPickedUp)
        {
            base.Update(gameTime);
        }
    }
}
