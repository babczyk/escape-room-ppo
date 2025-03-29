using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

public class Player : GameObject
{
    public Box heldBox;

    public Player(Vector2 position, Vector2 size, String roll = "PLAYER", float gravity = 9.8f, float terminalVelocity = 1000f)
        : base(position, size, roll, gravity, terminalVelocity) { }

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


    public new void Update(GameTime gameTime)
    {
        base.Update(gameTime);

        // Update held box position to follow the player
        if (heldBox != null)
        {
            heldBox.Position = new Vector2(Position.X + Size.X / 2 - heldBox.Size.X / 2 + 35, Position.Y - heldBox.Size.Y - 10);
        }
    }
}