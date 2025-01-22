using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

public class Player : GameObject
{
    private Box heldBox;

    public Player(Vector2 position, Vector2 size, float gravity = 9.8f, float terminalVelocity = 1000f)
        : base(position, size, gravity, terminalVelocity) { }

    public void Grab(Box box)
    {
        if (heldBox == null && Intersects(box) && !box.IsPickedUp)
        {
            heldBox = box;
            box.PickUp();
        }
    }

    public void DropHeldBox()
    {
        if (heldBox != null)
        {
            heldBox.Drop(Position + new Vector2(Size.X / 2, Size.Y));
            heldBox = null;
        }
    }

    public new void Update(GameTime gameTime, float groundLevel)
    {
        base.Update(gameTime, groundLevel);

        // Update held box position to follow the player
        if (heldBox != null)
        {
            heldBox.Position = new Vector2(Position.X + Size.X / 2 - heldBox.Size.X / 2, Position.Y - heldBox.Size.Y);
        }
    }
}