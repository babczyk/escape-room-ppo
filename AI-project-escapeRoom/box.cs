using System;
using System.Data;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
public class Box : Wall
{
    public bool IsPickedUp { get; private set; }

    public Box(Vector2 position, Vector2 size, String roll = "BOX") : base(position, size, roll) { }

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
